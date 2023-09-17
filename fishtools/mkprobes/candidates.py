from pathlib import Path
from typing import cast

import click
import polars as pl
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


def get_pseudogenes(
    gene: str, ensembl: ExternalData, y: SAMFrame, limit: int = 5
) -> tuple[pl.Series, pl.Series]:
    counts = (
        y.count("transcript")
        .join(
            ensembl.gtf[["transcript_id", "transcript_name"]],
            left_on="transcript",
            right_on="transcript_id",
        )
        .sort("count", descending=True)
    )

    # Filtered based on expression and number of probes aligned.
    ok = counts.filter(
        (pl.col("count") > 0.1 * pl.col("count").max())
        # & (pl.col("FPKM").lt(0.05 * pl.col("FPKM").first()) | pl.col("FPKM").lt(1) | pl.col("FPKM").is_null())
        & (
            pl.col("transcript_name").str.starts_with(gene + "-ps")
            | pl.col("transcript_name").str.starts_with("Gm")
        )
    )
    return ok[:limit]["transcript"], ok[:limit]["transcript_name"]


def get_candidates(
    dataset: Dataset,
    gene: str | None = None,
    transcript: str | None = None,
    output: str | Path = "output/",
    *,
    allow_pseudo: bool = True,
    ignore_revcomp: bool = False,
    realign: bool = False,
    allow: list[str] | None = None,
    disallow: list[str] | None = None,
):
    if not ((gene is None) ^ (transcript is None)):
        raise ValueError("Either gene or transcript must be specified.")

    if transcript is None:
        assert gene is not None
        # fmt: off
        transcript = get_transcripts(dataset, gene, mode="canonical")[0, "transcript_id"]
        appris = get_transcripts(dataset, gene, mode="appris")

        if not appris.filter(pl.col("transcript_id") == transcript).shape[0]:
            logger.warning(f"Ensembl canonical transcript {transcript} not found in APPRIS.")
            transcript = appris[0, "transcript_id"]

        logger.info(f"Chosen transcript: {transcript}")

    assert transcript is not None
    _run_transcript(
        dataset=dataset,
        transcript=transcript,
        output=output,
        allow_pseudo=allow_pseudo,
        ignore_revcomp=ignore_revcomp,
        realign=realign,
        allow=allow,
        disallow=disallow,
    )


def _run_bowtie(
    dataset: Dataset,
    seqs: pl.DataFrame,
    ignore_revcomp: bool = False,
):
    y = SAMFrame.from_bowtie_split_name(
        gen_fasta(seqs["seq"], names=seqs["name"]).getvalue(),
        dataset.path / "txome",
        seed_length=13,
        threshold=18,
        n_return=-1,
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


def _run_transcript(
    dataset: Dataset,
    transcript: str,
    output: str | Path = "output/",
    *,
    allow_pseudo: bool = True,
    ignore_revcomp: bool = False,
    realign: bool = False,
    allow: list[str] | None = None,
    disallow: list[str] | None = None,
):
    allow, disallow = allow or [], disallow or []
    (output := Path(output)).mkdir(exist_ok=True)

    try:
        row = dataset.ensembl.filter(
            pl.col("transcript_id" if transcript.startswith("ENS") else "transcript_name") == transcript
        ).to_dicts()[0]
    except IndexError:
        raise ValueError(f"Transcript {transcript} not found in ensembl.")
    del transcript

    gene, transcript_id, transcript_name = row["gene_name"], row["transcript_id"], row["transcript_name"]
    tss_gencode = set(dataset.gencode.filter(pl.col("gene_id") == row["gene_id"])["transcript_id"])
    tss_allofgene = set(dataset.ensembl.filter(pl.col("gene_id") == row["gene_id"])["transcript_id"])
    if not len(tss_gencode):
        logger.warning(f"Transcript {transcript_id} not found in GENCODE basic.")

    logger.info(f"Crawler running for {transcript_id}")

    if realign or not (output / f"{transcript_name}_all.parquet").exists():
        seq = dataset.gencode.get_seq(transcript_id, masked=True)
        crawled = crawler(seq, prefix=f"{gene}_{transcript_id}")
        y, offtargets = _run_bowtie(dataset, crawled, ignore_revcomp=ignore_revcomp)
        y.write_parquet(output / f"{transcript_name}_all.parquet")
        offtargets.write_parquet(output / f"{transcript_name}_offtargets.parquet")
    else:
        logger.warning(f"{transcript_name} reading from cache.")
        y = SAMFrame.read_parquet(output / f"{transcript_name}_all.parquet")
        offtargets = pl.read_parquet(output / f"{transcript_name}_offtargets.parquet")

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

    tss_pseudo, pseudo_name = (
        get_pseudogenes(gene, dataset.ensembl, y) if allow_pseudo else (pl.Series(), pl.Series())
    )
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
        y.filter_isin(transcript=tss_gencode)[["name", "transcript"]]
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

    ff = SAMFrame(
        y.filter_by_match(tss_allacceptable, match=0.8, match_consec=0.8)
        .agg_tm_offtarget(tss_allacceptable)
        .with_columns(
            transcript_name=pl.col("transcript").apply(dataset.ensembl.ts_to_gene),
            **PROBE_CRITERIA,
        )
        .with_columns(
            [
                (pl.col("gc_content").is_between(0.35, 0.65)).alias("ok_gc"),
                pl.col("seq").apply(lambda s: tm(cast(str, s), "rna", formamide=30)).alias("tm"),
                pl.col("seq").apply(lambda s: hp(cast(str, s), "rna", formamide=30)).alias("hp"),
            ]
        )
        .with_columns(oks=pl.sum(pl.col("^ok_.*$")))
        # .filter(~pl.col("seq").apply(lambda x: check_kmers(cast(str, x), dataset.kmerset, 18)))
        .filter(~pl.col("seq").apply(lambda x: dataset.check_kmers(cast(str, x))))
        .join(names_with_pseudo, on="name", how="left")
        .join(isoforms, on="name", how="left")
    )
    logger.info(f"Generated {len(ff)} candidates.")

    ff.write_parquet(output / f"{transcript_name}_crawled.parquet")
    return ff


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--gene", "-g", type=str, required=False, help="Gene name")
@click.option("--transcript", "-t", type=str, required=False, help="Transcript id")
@click.option("--output", "-o", type=click.Path(), default="output/")
@click.option("--ignore-revcomp", "-r", is_flag=True)
@click.option("--realign", is_flag=True)
def candidates(
    path: str,
    gene: str,
    transcript: str,
    output: str,
    ignore_revcomp: bool,
    realign: bool = False,
):
    """Initial screening of probes candidates for a gene."""
    get_candidates(
        Dataset(path),
        gene=gene,
        transcript=transcript,
        output=output,
        allow_pseudo=True,
        ignore_revcomp=ignore_revcomp,
        realign=realign,
    )
