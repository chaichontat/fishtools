# %%
from pathlib import Path
from typing import cast

import click
import polars as pl
from loguru import logger

from fishtools.ext.external_data import Dataset, ExternalData
from fishtools.mkprobes.alignment import gen_fasta
from fishtools.utils.geneframe import GeneFrame
from fishtools.utils.seqcalc import hp, tm

from ._crawler import crawler
from ._filtration import PROBE_CRITERIA, check_kmers

try:
    profile
except NameError:
    profile = lambda x: x


def get_pseudogenes(
    gene: str, ensembl: ExternalData, y: GeneFrame, limit: int = 5
) -> tuple[pl.Series, pl.Series]:
    counts = (
        y.count("transcript")
        # .left_join(fpkm, left_on="transcript", right_on="transcript_id(s)")
        .join(
            ensembl.gtf[["transcript_id", "transcript_name"]],
            left_on="transcript",
            right_on="transcript_id",
        ).sort("count", descending=True)
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


# %%
def run(
    dataset: Dataset,
    gene: str,
    output: str | Path,
    # overlap: int = -2,
    allow_pseudo: bool = True,
    ignore_revcomp: bool = False,
    realign: bool = False,
):
    (output := Path(output)).mkdir(exist_ok=True)
    tss_gencode = set(dataset.gencode.filter_gene(gene)["transcript_id"])
    tss_all = dataset.ensembl.filter_gene(gene)["transcript_id"]
    if not len(tss_gencode):
        raise ValueError(f"Gene {gene} not found in GENCODE.")

    canonical = dataset.ensembl.filter_gene(gene).filter(pl.col("tag") == "Ensembl_canonical")[
        0, "transcript_id"
    ]
    tss_gencode.add(canonical)  # Some canonical transcripts are not in GENCODE.
    logger.info(f"Crawler running for {gene}")
    logger.info(f"Canonical transcript: {canonical}")

    if realign or not (output / f"{gene}_all.parquet").exists():
        seq = dataset.gencode.get_seq(canonical)
        crawled = crawler(seq, prefix=f"{gene}_{canonical}")

        y = GeneFrame.from_bowtie_split_name(
            gen_fasta(crawled["seq"], names=crawled["name"]).getvalue(),
            dataset.path / "txome",
            seed_length=13,
            threshold=18,
            n_return=500,
            fasta=True,
            no_reverse=ignore_revcomp,
        )

        offtargets = (
            y["transcript"]
            .value_counts()
            .sort("counts", descending=True)
            .with_columns(name=pl.col("transcript").apply(dataset.ensembl.ts_to_gene))
        )

        y.write_parquet(output / f"{gene}_all.parquet")
        offtargets.write_parquet(output / f"{gene}_offtargets.parquet")
    else:
        logger.warning(f"{gene} reading from cache.")
        y = GeneFrame.read_parquet(output / f"{gene}_all.parquet")
        offtargets = pl.read_parquet(output / f"{gene}_offtargets.parquet")

    # Print most common offtargets
    logger.info(
        "Most common binders:\n"
        + str(
            y.count("transcript", descending=True)
            .filter(pl.col("count") > 0.1 * pl.col("count").first())
            .with_columns(name=pl.col("transcript").apply(dataset.ensembl.ts_to_tsname))[
                ["name", "transcript", "count"]
            ]
            # .join(dataset.fpkm, left_on="transcript", right_on="transcript_id(s)", how="left")
        )
    )

    tss_pseudo, pseudo_name = (
        get_pseudogenes(gene, dataset.ensembl, y) if allow_pseudo else (pl.Series(), pl.Series())
    )
    logger.info(f"Pseudogenes allowed: {', '.join(pseudo_name)}")
    pl.DataFrame(dict(transcript=[*tss_all, *tss_pseudo])).write_csv(output / f"{gene}_acceptable_tss.csv")

    if len(tss_pseudo):
        names_with_pseudo = (
            y.filter_isin(transcript=tss_pseudo)
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

    # ff = (
    #     y.filter_by_match([*tss_all, *tss_pseudo], match=0.8, match_consec=0.8).agg_tm_offtarget(
    #         [*tss_all, *tss_pseudo]
    #     )
    #     # .filter(pl.col("max_tm_offtarget") < 42)
    # )

    ff = GeneFrame(
        y.filter_by_match([*tss_all, *tss_pseudo], match=0.8, match_consec=0.8)
        .agg_tm_offtarget([*tss_all, *tss_pseudo])
        .lazy()
        .filter("is_ori_seq")
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
        .filter(~pl.col("seq").apply(lambda x: check_kmers(cast(str, x), dataset.kmerset, 18)))
        .filter(~pl.col("seq").apply(lambda x: check_kmers(cast(str, x), dataset.trna_rna_kmers, 15)))
        .collect()
        .join(names_with_pseudo, on="name", how="left")
        .join(isoforms, on="name", how="left")
    )
    # print(ff)
    # ff.unique("name").write_parquet(f"output/{gene}_filtered.parquet")
    # final = the_filter(ff, overlap=overlap).filter(pl.col("flag") & 16 == 0)
    # logger.info(final)
    ff.write_parquet(
        output
        / f"{gene}_crawled.parquet"
        # if overlap < 0
        # else output / f"{gene}_final_overlap_{overlap}.parquet"
    )
    return ff


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("gene")
@click.option("--output", "-o", type=click.Path(), default="output/")
@click.option("--debug", "-d", is_flag=True)
@click.option("--ignore-revcomp", "-r", is_flag=True)
@click.option("--realign", is_flag=True)
def candidates(
    path: str,
    gene: str,
    output: str,
    ignore_revcomp: bool,
    allow_pseudo: bool = True,
    debug: bool = False,
    realign: bool = False,
):
    """Initial screening of probes candidates for a gene."""
    run(
        Dataset(path),
        gene,
        output=output,
        allow_pseudo=allow_pseudo,
        ignore_revcomp=ignore_revcomp,
        realign=realign,
    )


if __name__ == "__main__":
    candidates()

    # sys.setrecursionlimit(5000)
    # pl.Config.set_fmt_str_lengths(100)
    # pl.Config.set_tbl_rows(100)

    # # %%
    # # GENCODE primary only
    # gtf = ExternalData(
    #     cache="data/mm39/gencode_vM32_transcripts.parquet",
    #     gtf_path="data/mm39/gencode.vM32.chr_patch_hapl_scaff.basic.annotation.gtf",
    #     fasta="data/mm39/combi.fa.gz",
    # )

    # gtf_all = ExternalData(
    #     cache="data/mm39/gencode_vM32_transcripts_all.parquet",
    #     gtf_path="data/mm39/Mus_musculus.GRCm39.109.gtf",
    #     fasta="data/mm39/combi.fa.gz",
    # )

    # fpkm = pl.read_parquet("data/fpkm/P0_combi.parquet")
    # kmer18 = pl.read_csv(
    #     "data/mm39/kmer_genome18.txt", separator=" ", has_header=False, new_columns=["kmer", "count"]
    # )
    # trna_rna_kmers = set(
    #     pl.read_csv(
    #         "data/mm39/kmer_trcombi15.txt", separator=" ", has_header=False, new_columns=["kmer", "count"]
    #     )["kmer"]
    # )
    # kmerset = set(kmer18["kmer"])

# %%