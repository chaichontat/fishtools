import json
import re
from pathlib import Path
from typing import cast

import click
import polars as pl
import pyfastx
from loguru import logger

from fishtools.mkprobes.constants import GOOD_SPECIES
from fishtools.mkprobes.genes.chkgenes import get_transcripts
from fishtools.mkprobes.starmap.starmap import split_probe

from .ext.dataset import Dataset, ReferenceDataset
from .ext.external_data import ExternalData
from .utils._alignment import gen_fasta
from .utils._crawler import crawler
from .utils._filtration import PROBE_CRITERIA, visualize_probe_coverage
from .utils.samframe import SAMFrame
from .utils.seqcalc import hp, tm

try:
    profile  # type: ignore
except NameError:
    profile = lambda x: x


pl.Config.set_tbl_rows(30)
EXTRA_FLAG = "-extra"


def get_pseudogenes(
    genes: list[str],
    ensembl: ExternalData,
    y: SAMFrame,
    allow_pseudogenes: bool = False,
    limit: int = -1,  # allow: list[str] | None = None
) -> tuple[pl.Series, pl.DataFrame]:
    counts = (
        SAMFrame(y)
        .count_group_by("transcript")
        .join(
            ensembl.gtf[["transcript_id", "transcript_name"]],
            left_on="transcript",
            right_on="transcript_id",
        )
        .sort("count", descending=True)
        .with_row_count("i")
        .with_columns(
            is_related=pl.col("transcript_name").str.contains(rf"^({'|'.join(genes)})-[\w-]?.*")
            | pl.col("transcript").is_in(genes),
            significant=pl.col("count") > 0.1 * pl.col("count").max(),
        )
    )
    max_related = counts.filter(pl.col("is_related"))["i"].max()

    # print(genes, max_related, counts)
    counts = counts.with_columns(
        maybe_acceptable=(pl.col("transcript_name").str.starts_with("Gm"))
        & (pl.col("count") > 0.5 * pl.col("count").max())
        & pl.col("i").lt(max_related),
        # | pl.col("i").lt(max_related)
        # & (
        #     pl.col("transcript_name").is_null()
        #     | pl.col("transcript_name").str.starts_with("Gm")
        # )
    ).with_columns(
        acceptable=pl.col("is_related")
        if not allow_pseudogenes
        else (pl.col("is_related") | pl.col("maybe_acceptable"))
    )

    not_acceptable = counts.filter(~pl.col("acceptable"))
    # cutoff when the probes bind to other genes
    # limit = limit if limit > 0 else (not_acceptable[0, "i"] - 1 or 10)

    if len(not_acceptable) > 0.1 * len(counts):
        logger.warning(f"More than 10% of candidates of {genes} bind to other genes.")
        # print(not_acceptable[:50])

    # Filter based on expression and number of probes aligned.
    return counts.filter(pl.col("significant") & pl.col("acceptable"))["transcript"], counts


def get_candidates(
    dataset: Dataset | ReferenceDataset,
    transcript: str | None = None,
    seq: str | None = None,
    output: str | Path = "output/",
    *,
    ignore_revcomp: bool = False,
    overwrite: bool = False,
    allow: list[str] | None = None,
    disallow: list[str] | None = None,
    pseudogene_limit: int = -1,
):
    if not ((transcript is not None) ^ (seq is not None)):
        raise ValueError("Either gene or sequence must be specified.")

    # if (
    #     seq is None
    #     and transcript
    #     and not (
    #         re.search(r"\-2\d\d", transcript)
    #         or "ENSMUST" in transcript
    #         or "ENST" in transcript
    #     )
    #     and dataset.ensembl is not None
    # ):
    #     transcript = get_transcripts(dataset, [transcript], mode="canonical")[
    #         0, "transcript_id"
    #     ]
    #     appris = get_transcripts(dataset, [transcript], mode="appris")

    #     if not appris.filter(pl.col("transcript_id") == transcript).shape[0]:
    #         logger.warning(
    #             f"Ensembl canonical transcript {transcript} not found in APPRIS."
    #         )
    #         transcript = appris[0, "transcript_id"]

    #     logger.info(f"Chosen transcript: {transcript}")
    # else:
    transcript = transcript

    if isinstance(dataset, ReferenceDataset):
        _run_transcript(
            dataset=dataset,
            transcript=transcript,
            fasta=seq,
            output=output,
            ignore_revcomp=ignore_revcomp,
            overwrite=overwrite,
            allow=allow,
            disallow=disallow,
            pseudogene_limit=pseudogene_limit,
        )
    else:
        _run_transcript_generic(
            dataset=dataset,
            transcript=transcript,
            output=output,
            ignore_revcomp=ignore_revcomp,
            overwrite=overwrite,
            allow=allow,
            disallow=disallow,
        )


def _run_bowtie(
    dataset: Dataset,
    seqs: pl.DataFrame,
    ignore_revcomp: bool = False,
    bowtie2_index: str | Path | None = None,
    **kwargs,
):
    y = SAMFrame.from_bowtie_split_name(
        gen_fasta(seqs["seq"], names=seqs["name"]).getvalue(),
        bowtie2_index if bowtie2_index else dataset.data.bowtie2_index,
        seed_length=12,
        threshold=16,
        n_return=200,
        fasta=True,
        no_reverse=ignore_revcomp,
        capture_stderr=True,
        transcript_regex=dataset.data.key_regex,  # type: ignore
        **kwargs,
    )
    df = (
        y["transcript"]
        .value_counts()
        .sort("count", descending=True)
        .with_columns(
            name=pl.when(pl.col("transcript").ne("*"))
            .then(pl.col("transcript").map_elements(dataset.ensembl.ts_to_gene, return_dtype=pl.Utf8))
            .otherwise(pl.col("transcript"))
            if isinstance(dataset, ReferenceDataset)
            else pl.col("transcript")
        )
    )

    return y, df


def _convert_gene_to_tss(dataset: Dataset, gtss: list[str]):
    out = []
    for gts in gtss:
        if gts.startswith("ENS"):
            out.append(gts)
            continue
        out.extend(get_transcripts(dataset, [gts], "canonical")["transcript_id"].to_list())
    return out


def _run_transcript(
    dataset: ReferenceDataset,
    transcript: str | None = None,
    fasta: str | Path | None = None,
    output: str | Path = "output/",
    *,
    ignore_revcomp: bool = False,
    overwrite: bool = False,
    allow: list[str] | None = None,
    disallow: list[str] | None = None,
    formamide: int = 40,
    pseudogene_limit: int = -1,
):
    allow, disallow = allow or [], disallow or []

    allow_tss = _convert_gene_to_tss(dataset, allow)
    disallow_tss = _convert_gene_to_tss(dataset, disallow)

    (output := Path(output)).mkdir(exist_ok=True)

    if not ((fasta is not None) ^ (transcript is not None)):
        raise ValueError("Either FASTA or transcript must be specified.")

    if fasta is not None:
        raise NotImplementedError("FASTA not implemented yet.")

    if transcript is not None and not transcript.endswith(EXTRA_FLAG):
        transcript = transcript.split(".")[0]
        try:
            row = dataset.ensembl.filter(
                pl.col("transcript_id" if transcript.startswith("ENS") else "transcript_name") == transcript
            ).to_dicts()[0]
        except IndexError:
            raise ValueError(f"Transcript {transcript} not found in ensembl.")

        gene, transcript_id, transcript_name = (
            row["gene_name"],
            row["transcript_id"],
            row["transcript_name"],
        )
        # need to be gene_name, not id, because of genes on scaffold assemblies
        tss_gencode = set(dataset.data.filter(pl.col("gene_name") == row["gene_name"])["transcript_id"])
        tss_allofgene = set(dataset.ensembl.filter(pl.col("gene_name") == row["gene_name"])["transcript_id"])
        if not len(tss_gencode):
            logger.warning(f"Transcript {transcript_id} not found in GENCODE basic.")

        logger.info(f"Crawler running for {transcript_id}")
    elif transcript is not None and transcript.endswith(EXTRA_FLAG):
        transcript_id = transcript_name = gene = transcript = transcript
        tss_gencode = tss_allofgene = {transcript}
    else:
        tss_gencode = tss_allofgene = set()
        gene, seq = next(pyfastx.Fastx(fasta))
        transcript_id = transcript_name = gene
    del transcript

    gene: str
    transcript_id: str
    transcript_name: str
    try:
        if overwrite:
            raise FileNotFoundError
        offtargets = pl.read_parquet(output / f"{transcript_name}_bowtie.parquet")
        y = pl.read_parquet(output / f"{transcript_name}_all.parquet")
        try:
            stats = json.loads(output.joinpath(f"{transcript_name}_crawled.stats.json").read_text())
        except FileNotFoundError:
            stats = {}
    except (FileNotFoundError, pl.exceptions.ComputeError):
        seq = dataset.data.get_seq(transcript_id, convert=dataset.species in GOOD_SPECIES)
        if len(seq) < 1500:
            logger.warning(
                f"Transcript {transcript_name} is only {len(seq)}bp long. There may not be enough probes."
            )
        else:
            logger.info(f"{transcript_name}: {len(seq)}bp")

        crawled = crawler(
            seq,
            prefix=f"{gene if fasta is None else ''}_{transcript_id}",
            formamide=formamide,
        )
        # visualize_probe_coverage(
        #     crawled["pos_start"],
        #     crawled["pos_end"],
        #     len(seq),
        #     output / f"{transcript_name}_crawled.coverage.txt",
        # )
        if len(crawled) > 5000:
            logger.warning(f"Transcript {transcript_id} has {len(crawled)} probes. Using only 2000.")
            crawled = crawled.sample(n=5000, shuffle=True, seed=3)
        if len(crawled) < 5:
            logger.warning(f"Transcript {transcript_id} has only {len(crawled)} probes.")
        if len(crawled) < 100:
            logger.warning(f"Transcript {transcript_id} has only {len(crawled)} probes.")

        crawled = (
            crawled.with_columns(
                splitted=pl.col("seq").map_elements(
                    lambda pos: split_probe(pos, 60), return_dtype=pl.List(pl.Utf8)
                ),
                seq_full=pl.col("seq"),
            )
            .with_columns(
                # need to be swapped because split_probe is not rced.
                splint=pl.col("splitted").list.get(1),
                padlock=pl.col("splitted").list.get(0),
                pad_start=pl.col("splitted").list.get(2).cast(pl.Int16),
            )
            .drop("splitted")
            .filter(pl.col("splint").str.len_chars() > 0)
        )
        # To get pad_start
        crawled = (
            crawled.melt(
                id_vars=[col for col in crawled.columns if col not in ["splint", "padlock"]],
                value_vars=["splint", "padlock"],
            )
            .with_columns(seq=pl.col("value"), name=pl.col("name") + "_" + pl.col("variable"))
            .drop(["variable", "value"])
        )
        # crawled.write_parquet(output / f"{transcript_name}_rawcrawled.parquet")

        y, offtargets = _run_bowtie(dataset, crawled, ignore_revcomp=ignore_revcomp)
        y = y.join(crawled[["name", "seq_full", "pad_start"]], on="name")

        y.write_parquet(output / f"{transcript_name}_all.parquet")
        offtargets.write_parquet(output / f"{transcript_name}_bowtie.parquet")
        stats = {
            "seq_length": len(seq),
            "crawled_length": len(crawled),
        }

    # Print most common offtargets
    logger.info(
        "Most common binders:\n"
        + str(
            SAMFrame(y)
            .count_group_by("transcript", descending=True)
            .filter(pl.col("count") > 0.1 * pl.col("count").first())
            .with_columns(
                name=pl.col("transcript").map_elements(dataset.ensembl.ts_to_tsname, return_dtype=pl.Utf8)
            )[["name", "transcript", "count"]]
        )
    )

    if dataset.species in GOOD_SPECIES:
        tss_pseudo, counts = (
            get_pseudogenes(list({gene} | set(allow)), dataset.ensembl, y, limit=pseudogene_limit)
            if pseudogene_limit
            else (pl.Series(), pl.DataFrame())
        )
        if len(bad_sig := counts.filter(~pl.col("acceptable") & pl.col("significant"))):
            bad_sig.write_csv(output / f"{transcript_name}_offtarget_counts.csv")

        # logger.info(f"Pseudogenes allowed: {', '.join(pseudo_name)}")
        tss_others = {*allow_tss} - set(disallow_tss)

        if len(tss_others):
            names_with_pseudo = (
                SAMFrame(y)
                .filter_isin(transcript=tss_others)
                .rename(dict(transcript="maps_to_pseudo"))[["name", "maps_to_pseudo"]]
                .unique("name")
            )
        else:
            names_with_pseudo = pl.DataFrame({
                "name": y["name"].unique(),
                "maps_to_pseudo": "",
            })

        isoforms = (
            SAMFrame(y)
            .filter_isin(transcript=tss_allofgene)[["name", "transcript"]]
            .with_columns(
                isoforms=pl.col("transcript").map_elements(dataset.ensembl.ts_to_tsname, return_dtype=pl.Utf8)
            )[["name", "isoforms"]]
            .group_by("name")
            .all()
        )

        tss_allacceptable: list[str] = list(tss_allofgene | tss_others) + ["*"]
        pl.DataFrame(
            dict(
                transcript_id=tss_allacceptable,
                transcript_name=[dataset.ensembl.ts_to_tsname(t) for t in tss_allacceptable if t != "*"]
                + ["*"],
            )
        ).sort("transcript_name").write_csv(output / f"{transcript_name}_acceptable_tss.csv")
    else:
        tss_allacceptable = allow

    ff = SAMFrame(y).filter_by_match(tss_allacceptable, match=0.8, match_consec=0.8)
    if not len(ff):
        raise Exception(
            f"No probes left after filtering for {transcript_name}. Most likely, there is a homologous gene involved."
        )

    ff = SAMFrame(
        ff.agg_tm_offtarget(tss_allacceptable, formamide=formamide)
        .with_columns(
            transcript_name=pl.col("transcript").map_elements(
                dataset.ensembl.ts_to_gene, return_dtype=pl.Utf8
            ),
            **PROBE_CRITERIA,
        )
        .with_columns([
            (pl.col("gc_content").is_between(0.35, 0.65)).alias("ok_gc"),
            pl.col("seq")
            .map_elements(
                lambda s: tm(cast(str, s), "hybrid", formamide=formamide),
                return_dtype=pl.Float32,
            )
            .alias("tm"),
            pl.col("seq")
            .map_elements(
                lambda s: hp(cast(str, s), "hybrid", formamide=formamide),
                return_dtype=pl.Float32,
            )
            .alias("hp"),
        ])
        .with_columns(oks=pl.sum_horizontal(pl.col("^ok_.*$")))
        # .filter(~pl.col("seq").map_elements(lambda x: check_kmers(cast(str, x), dataset.kmerset, 18)))
        .filter(
            ~pl.col("seq").map_elements(lambda x: dataset.check_kmers(cast(str, x)), return_dtype=pl.Boolean)
        )
    )
    notoks = ff.filter(pl.col("oks") == 0)
    if not notoks.is_empty():
        logger.info(f"{len(notoks)} have no priority")

    if dataset.species in GOOD_SPECIES:
        ff = ff.join(names_with_pseudo, on="name", how="left").join(isoforms, on="name", how="left")

    logger.info(f"Generated {len(ff)} candidates.")

    assert not ff["seq"].str.contains("N").any()
    stats |= {
        "allow": list(tss_allacceptable),
        "offtargets": offtargets.to_dicts(),
        "post_match_filter": len(ff),
    }
    ff.write_parquet(output / f"{transcript_name}_crawled.parquet")
    output.joinpath(f"{transcript_name}_crawled.stats.json").write_text(json.dumps(stats, indent=2))
    return ff


def _run_transcript_generic(
    dataset: Dataset,
    transcript: str,
    output: str | Path = "output/",
    *,
    ignore_revcomp: bool = False,
    overwrite: bool = False,
    allow: list[str] | None = None,
    disallow: list[str] | None = None,
    formamide: int = 40,
):
    allow, disallow = allow or [], disallow or []
    (output := Path(output)).mkdir(exist_ok=True)

    try:
        if overwrite:
            raise FileNotFoundError
        offtargets = pl.read_parquet(output / f"{transcript}_bowtie.parquet")
        y = pl.read_parquet(output / f"{transcript}_all.parquet")
        try:
            stats = json.loads(output.joinpath(f"{transcript}_crawled.stats.json").read_text())
        except FileNotFoundError:
            stats = {}
    except (FileNotFoundError, pl.exceptions.ComputeError):
        seq = dataset.data.get_seq(transcript, convert=False)
        if len(seq) < 1500:
            logger.warning(
                f"Transcript {transcript} is only {len(seq)}bp long. There may not be enough probes."
            )
        else:
            logger.info(f"{transcript}: {len(seq)}bp")

        crawled = crawler(
            seq,
            prefix=f"{transcript}_{transcript}",
            formamide=formamide,
            length_limit=(43, 54),
        )
        if len(crawled) > 5000:
            logger.warning(f"Transcript {transcript} has {len(crawled)} probes. Using only 2000.")
            crawled = crawled.sample(n=5000, shuffle=True, seed=3)
        if len(crawled) < 5:
            logger.warning(f"Transcript {transcript} has only {len(crawled)} probes.")
        if len(crawled) < 100:
            logger.warning(f"Transcript {transcript} has only {len(crawled)} probes.")

        crawled = (
            crawled.with_columns(
                splitted=pl.col("seq").map_elements(
                    lambda pos: split_probe(pos, 60), return_dtype=pl.List(pl.Utf8)
                ),
                seq_full=pl.col("seq"),
            )
            .with_columns(
                # need to be swapped because split_probe is not rced.
                splint=pl.col("splitted").list.get(1),
                padlock=pl.col("splitted").list.get(0),
                pad_start=pl.col("splitted").list.get(2).cast(pl.Int16),
            )
            .drop("splitted")
            .filter(pl.col("splint").str.len_chars() > 0)
        )
        # print(crawled.schema)
        visualize_probe_coverage(
            crawled["pos_start"],
            crawled["pos_end"],
            len(seq),
            output_file=output / f"{transcript}_crawled.coverage.txt",
        )
        # To get pad_start
        crawled = (
            crawled.melt(
                id_vars=[col for col in crawled.columns if col not in ["splint", "padlock"]],
                value_vars=["splint", "padlock"],
            )
            .with_columns(seq=pl.col("value"), name=pl.col("name") + "_" + pl.col("variable"))
            .drop(["variable", "value"])
        )
        # crawled.write_parquet(output / f"{transcript_name}_rawcrawled.parquet")

        y, offtargets = _run_bowtie(dataset, crawled, ignore_revcomp=ignore_revcomp)
        y = y.join(crawled[["name", "seq_full", "pad_start"]], on="name").with_columns(
            maps_to_pseudo=pl.lit("")
        )

        y.write_parquet(output / f"{transcript}_all.parquet")
        offtargets.write_parquet(output / f"{transcript}_bowtie.parquet")
        stats = {
            "seq_length": len(seq),
            "crawled_length": len(crawled),
        }

    # Print most common offtargets
    logger.info(
        "Most common binders:\n"
        + str(
            offtargets := SAMFrame(y)
            .count_group_by("transcript", descending=True)
            .filter(pl.col("count") > 0.1 * pl.col("count").first())
            # .with_columns(
            #     name=pl.col("transcript").map_elements(dataset.ensembl.ts_to_tsname, return_dtype=pl.Utf8)
            # )[["name", "transcript", "count"]]
        )
    )

    tss_allacceptable = set(allow + [transcript])

    ff = SAMFrame(y).filter_by_match(tss_allacceptable, match=0.8, match_consec=0.8)
    if not len(ff):
        raise Exception(
            f"No probes left after filtering for {transcript}. Most likely, there is a homologous gene involved."
        )

    ff = SAMFrame(
        ff.agg_tm_offtarget(tss_allacceptable, formamide=formamide)
        .with_columns(
            # transcript_name=pl.col("transcript").map_elements(
            #     dataset.ensembl.ts_to_gene, return_dtype=pl.Utf8
            # ),
            **PROBE_CRITERIA,
        )
        .with_columns([
            (pl.col("gc_content").is_between(0.35, 0.65)).alias("ok_gc"),
            pl.col("seq")
            .map_elements(
                lambda s: tm(cast(str, s), "hybrid", formamide=formamide),
                return_dtype=pl.Float32,
            )
            .alias("tm"),
            pl.col("seq")
            .map_elements(
                lambda s: hp(cast(str, s), "hybrid", formamide=formamide),
                return_dtype=pl.Float32,
            )
            .alias("hp"),
        ])
        .with_columns(oks=pl.sum_horizontal(pl.col("^ok_.*$")))
        # .filter(~pl.col("seq").map_elements(lambda x: check_kmers(cast(str, x), dataset.kmerset, 18)))
    )

    logger.info(f"Generated {len(ff)} candidates.")

    assert not ff["seq"].str.contains("N").any()
    stats |= {
        "allow": list(tss_allacceptable),
        "offtargets": offtargets.to_dicts(),
        "post_match_filter": len(ff),
    }

    ff.write_parquet(output / f"{transcript}_crawled.parquet")
    output.joinpath(f"{transcript}_crawled.stats.json").write_text(json.dumps(stats, indent=2))
    return ff, stats


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--gene", "-g", type=str, required=False, help="Gene name")
@click.option(
    "--fasta",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.option("--output", "-o", type=click.Path(), default="output/")
@click.option("--ignore-revcomp", "-r", is_flag=True)
@click.option("--overwrite", is_flag=True)
@click.option("--pseudogene-limit", type=int, default=-1)
@click.option(
    "--allow",
    type=str,
    help="Don't filter out probes that bind to these genes, separated by comma.",
)
@click.option(
    "--disallow",
    type=str,
    help="DO filter out probes that bind to these genes, separated by comma.",
)
def candidates(
    path: str,
    gene: str | None,
    fasta: Path | None,
    output: str | Path = "output/",
    ignore_revcomp: bool = False,
    overwrite: bool = False,
    allow: str | None = None,
    disallow: str | None = None,
    pseudogene_limit: int = -1,
):
    """Initial screening of probes candidates for a gene."""
    allow_ = allow.split(",") if allow else None
    disallow_ = disallow.split(",") if disallow else None
    get_candidates(
        Dataset(path),
        transcript=gene,
        seq=fasta,
        output=output,
        ignore_revcomp=ignore_revcomp,
        overwrite=overwrite,
        allow=allow_,
        disallow=disallow_,
        pseudogene_limit=pseudogene_limit,
    )
