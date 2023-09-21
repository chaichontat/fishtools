import re
from pathlib import Path
from typing import cast

import click
import polars as pl
import pyfastx
from loguru import logger

from fishtools.mkprobes.genes.chkgenes import get_transcripts

from .ext.external_data import Dataset, ExternalData
from .utils._alignment import gen_fasta
from .utils._crawler import crawler
from .utils._filtration import PROBE_CRITERIA
from .utils.samframe import SAMFrame
from .utils.seqcalc import hp, tm

try:
    profile  # type: ignore
except NameError:
    profile = lambda x: x


pl.Config.set_tbl_rows(30)


def get_pseudogenes(
    gene: str, ensembl: ExternalData, y: SAMFrame, limit: int = -1
) -> tuple[pl.Series, pl.DataFrame]:
    counts = (
        y.count("transcript")
        .join(
            ensembl.gtf[["transcript_id", "transcript_name"]],
            left_on="transcript",
            right_on="transcript_id",
        )
        .sort("count", descending=True)
        .with_row_count("i")
        .with_columns(
            acceptable=pl.col("transcript_name").str.contains(rf"^{gene}.*")
            | pl.col("transcript_name").str.starts_with("Gm")
            | pl.col("transcript_name").is_null(),
            significant=pl.col("count") > 0.1 * pl.col("count").max(),
        )
    )

    not_acceptable = counts.filter(~pl.col("acceptable"))
    # cutoff when the probes bind to other genes
    limit = limit if limit > 0 else (not_acceptable[0, "i"] - 1 or 10)

    if len(not_acceptable):
        logger.warning(f"More than 10% of candidates of {gene} bind to other genes.")
        print(not_acceptable[:50])

    # Filter based on expression and number of probes aligned.
    return counts.filter(pl.col("significant") & pl.col("acceptable"))[:limit]["transcript"], counts


def get_candidates(
    dataset: Dataset,
    gene: str | None = None,
    fasta: Path | str | None = None,
    output: str | Path = "output/",
    *,
    ignore_revcomp: bool = False,
    realign: bool = False,
    allow: list[str] | None = None,
    disallow: list[str] | None = None,
    pseudogene_limit: int = -1,
):
    if not ((gene is not None) ^ (fasta is not None)):
        raise ValueError("Either gene or FASTA must be specified.")

    if fasta is None and gene and not (re.search(r"\-2\d\d", gene) or "ENSMUST" in gene or "ENST" in gene):
        # fmt: off
        transcript = get_transcripts(dataset, gene, mode="canonical")[0, "transcript_id"]
        appris = get_transcripts(dataset, gene, mode="appris")

        if not appris.filter(pl.col("transcript_id") == transcript).shape[0]:
            logger.warning(f"Ensembl canonical transcript {transcript} not found in APPRIS.")
            transcript = appris[0, "transcript_id"]

        logger.info(f"Chosen transcript: {transcript}")
    else:
        transcript = gene

    _run_transcript(
        dataset=dataset,
        transcript=transcript,
        fasta=fasta,
        output=output,
        ignore_revcomp=ignore_revcomp,
        realign=realign,
        allow=allow,
        disallow=disallow,
        pseudogene_limit=pseudogene_limit,
    )


def _run_bowtie(
    dataset: Dataset,
    seqs: pl.DataFrame,
    ignore_revcomp: bool = False,
):
    y = SAMFrame.from_bowtie_split_name(
        gen_fasta(seqs["seq"], names=seqs["name"]).getvalue(),
        dataset.path / "txome",
        seed_length=12,
        threshold=16,
        n_return=200,
        fasta=True,
        no_reverse=ignore_revcomp,
    )

    return (
        y,
        y["transcript"]
        .value_counts()
        .sort("counts", descending=True)
        .with_columns(name=pl.col("transcript").apply(dataset.ensembl.ts_to_gene)),
    )


def _convert_gene_to_tss(dataset: Dataset, gtss: list[str]):
    out = []
    for gts in gtss:
        if gts.startswith("ENS"):
            out.append(gts)
            continue
        out.extend(get_transcripts(dataset, gts, "ensembl")["transcript_id"].to_list())
    return out


def _run_transcript(
    dataset: Dataset,
    transcript: str | None = None,
    fasta: str | Path | None = None,
    output: str | Path = "output/",
    *,
    ignore_revcomp: bool = False,
    realign: bool = False,
    allow: list[str] | None = None,
    disallow: list[str] | None = None,
    formamide: int = 45,
    pseudogene_limit: int = -1,
):
    allow, disallow = allow or [], disallow or []
    allow = _convert_gene_to_tss(dataset, allow)
    disallow = _convert_gene_to_tss(dataset, disallow)

    (output := Path(output)).mkdir(exist_ok=True)

    if not ((fasta is not None) ^ (transcript is not None)):
        raise ValueError("Either FASTA or transcript must be specified.")

    if transcript is not None:
        transcript = transcript.split(".")[0]
        try:
            row = dataset.ensembl.filter(
                pl.col("transcript_id" if transcript.startswith("ENS") else "transcript_name") == transcript
            ).to_dicts()[0]
        except IndexError:
            raise ValueError(f"Transcript {transcript} not found in ensembl.")

        gene, transcript_id, transcript_name = row["gene_name"], row["transcript_id"], row["transcript_name"]
        # need to be gene_name, not id, because of genes on scaffold assemblies
        tss_gencode = set(dataset.gencode.filter(pl.col("gene_name") == row["gene_name"])["transcript_id"])
        tss_allofgene = set(dataset.ensembl.filter(pl.col("gene_name") == row["gene_name"])["transcript_id"])
        if not len(tss_gencode):
            logger.warning(f"Transcript {transcript_id} not found in GENCODE basic.")

        logger.info(f"Crawler running for {transcript_id}")

    else:
        tss_gencode = tss_allofgene = set()
        gene, seq = next(pyfastx.Fastx(fasta))
        transcript_id = transcript_name = gene

    gene: str
    transcript_id: str
    transcript_name: str

    if realign or not (output / f"{transcript_name}_bowtie.parquet").exists():
        seq = dataset.gencode.get_seq(transcript_id) if fasta is None else seq
        if len(seq) < 1500:
            logger.warning(
                f"Transcript {transcript_name} is only {len(seq)}bp long. There may not be enough probes."
            )
        else:
            logger.info(f"{transcript_name}: {len(seq)}bp")

        crawled = crawler(seq, prefix=f"{gene if fasta is None else ''}_{transcript_id}", formamide=formamide)
        if len(crawled) > 5000:
            logger.warning(f"Transcript {transcript_id} has {len(crawled)} probes. Using only 2000.")
            crawled = crawled.sample(n=2000, shuffle=True, seed=3)
        if len(crawled) < 50:
            raise Exception(f"Transcript {transcript_id} has only {len(crawled)} probes.")
        if len(crawled) < 100:
            logger.warning(f"Transcript {transcript_id} has only {len(crawled)} probes.")

        y, offtargets = _run_bowtie(dataset, crawled, ignore_revcomp=ignore_revcomp)
        y.write_parquet(output / f"{transcript_name}_all.parquet")
        offtargets.write_parquet(output / f"{transcript_name}_bowtie.parquet")
    else:
        logger.warning(f"{transcript_name} reading from cache.")
        y = SAMFrame.read_parquet(output / f"{transcript_name}_all.parquet")
        offtargets = pl.read_parquet(output / f"{transcript_name}_bowtie.parquet")

    # Print most common offtargets
    logger.info(
        "Most common binders:\n"
        + str(
            y.count("transcript", descending=True)
            .filter(pl.col("count") > 0.1 * pl.col("count").first())
            .with_columns(name=pl.col("transcript").apply(dataset.ensembl.ts_to_tsname))[
                ["name", "transcript", "count"]
            ]
        )
    )

    tss_pseudo, counts = (
        get_pseudogenes(gene, dataset.ensembl, y, limit=pseudogene_limit)
        if pseudogene_limit
        else (pl.Series(), pl.DataFrame())
    )

    if len(bad_sig := counts.filter(~pl.col("acceptable") & pl.col("significant"))):
        bad_sig.write_csv(output / f"{transcript_name}_offtarget_counts.csv")

    # logger.info(f"Pseudogenes allowed: {', '.join(pseudo_name)}")
    tss_others = {*tss_pseudo, *allow} - set(disallow)

    if len(tss_others):
        names_with_pseudo = (
            y.filter_isin(transcript=tss_others)
            .rename(dict(transcript="maps_to_pseudo"))[["name", "maps_to_pseudo"]]
            .unique("name")
        )
    else:
        names_with_pseudo = pl.DataFrame({"name": y["name"].unique(), "maps_to_pseudo": ""})

    isoforms = (
        y.filter_isin(transcript=tss_allofgene)[["name", "transcript"]]
        .with_columns(isoforms=pl.col("transcript").apply(dataset.ensembl.ts_to_tsname))[["name", "isoforms"]]
        .groupby("name")
        .all()
    )

    tss_allacceptable: list[str] = list(tss_allofgene | tss_others)
    pl.DataFrame(
        dict(
            transcript_id=tss_allacceptable,
            transcript_name=[dataset.ensembl.ts_to_tsname(t) for t in tss_allacceptable],
        )
    ).sort("transcript_name").write_csv(output / f"{transcript_name}_acceptable_tss.csv")

    ff = y.filter_by_match(tss_allacceptable, match=0.8, match_consec=0.8)
    if not len(ff):
        raise Exception(
            f"No probes left after filtering for {transcript_name}. Most likely, there is a homologous gene involved."
        )

    ff = SAMFrame(
        ff.agg_tm_offtarget(tss_allacceptable, formamide=formamide)
        .with_columns(
            transcript_name=pl.col("transcript").apply(dataset.ensembl.ts_to_gene),
            **PROBE_CRITERIA,
        )
        .with_columns(
            [
                (pl.col("gc_content").is_between(0.35, 0.65)).alias("ok_gc"),
                pl.col("seq").apply(lambda s: tm(cast(str, s), "rna", formamide=formamide)).alias("tm"),
                pl.col("seq").apply(lambda s: hp(cast(str, s), "rna", formamide=formamide)).alias("hp"),
            ]
        )
        .with_columns(oks=pl.sum(pl.col("^ok_.*$")))
        # .filter(~pl.col("seq").apply(lambda x: check_kmers(cast(str, x), dataset.kmerset, 18)))
        .filter(~pl.col("seq").apply(lambda x: dataset.check_kmers(cast(str, x))))
        .join(names_with_pseudo, on="name", how="left")
        .join(isoforms, on="name", how="left")
    )
    logger.info(f"Generated {len(ff)} candidates.")

    assert not ff["seq"].str.contains("N").any()
    ff.write_parquet(output / f"{transcript_name}_crawled.parquet")
    return ff


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--gene", "-g", type=str, required=False, help="Gene name")
@click.option("--fasta", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(), default="output/")
@click.option("--ignore-revcomp", "-r", is_flag=True)
@click.option("--realign", is_flag=True)
@click.option("--pseudogene-limit", type=int, default=-1)
@click.option("--allow", type=str, multiple=True, help="Don't filter out probes that bind to these genes")
@click.option("--disallow", type=str, multiple=True, help="DO filter out probes that bind to these genes")
def candidates(
    path: str,
    gene: str | None,
    fasta: Path | None,
    output: str | Path = "output/",
    ignore_revcomp: bool = True,
    realign: bool = False,
    allow: list[str] | None = None,
    disallow: list[str] | None = None,
    pseudogene_limit: int = -1,
):
    """Initial screening of probes candidates for a gene."""
    get_candidates(
        Dataset(path),
        gene=gene,
        fasta=fasta,
        output=output,
        ignore_revcomp=ignore_revcomp,
        realign=realign,
        allow=allow,
        disallow=disallow,
        pseudogene_limit=pseudogene_limit,
    )
