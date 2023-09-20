# %%
import json
import re
from itertools import cycle, permutations
from pathlib import Path
from typing import Final, Iterable, Sequence, cast

import click
import polars as pl
from loguru import logger

from fishtools.mkprobes.candidates import _run_bowtie
from fishtools.mkprobes.ext.external_data import Dataset

READOUTS: Final[dict[int, str]] = {
    x["id"]: x["seq"] for x in pl.read_csv(Path(__file__).parent / "readout_ref_filtered.csv").to_dicts()
}


def assign_overlap(
    output: Path | str, gene: str, *, min_probes: int = 64, target_probes: int = 72, max_overlap: int = 20
) -> int:
    if max_overlap % 5 != 0 or max_overlap < 0:
        raise ValueError("max_overlap must be a multiple of 5")
    output = Path(output)
    for ol in range(0, max_overlap + 1, 5):
        df = pl.read_parquet(
            f"output/{gene}_screened_ol{ol}.parquet" if ol else f"output/{gene}_screened.parquet"
        )
        if len(df) >= target_probes:
            return ol

    # if len(df) >= min_probes:  # type: ignore
    return ol  # type: ignore

    # raise ValueError(f"Gene {gene} cannot be fixed")


def stitch(seq: str, codes: Sequence[int], sep: str = "TT") -> str:
    return READOUTS[codes[0]] + sep + seq + sep + READOUTS[codes[1]]


def construct_encoding(seq_encoding: pl.DataFrame, idxs: list[int]):
    if any(idx not in READOUTS for idx in idxs):
        raise ValueError(f"Invalid readout indices: {idxs}")

    pairs = cast(
        Iterable[tuple[int, int]],
        cycle(permutations(idxs, 2)) if len(idxs) > 1 else cycle([(idxs[0], idxs[0])]),
    )

    out = dict(name=[], seq=[], code1=[], code2=[])

    for (name, seq), codes in zip(seq_encoding[["name", "seq"]].iter_rows(), pairs):
        for sep in ["TAA", "TAT", "TTA", "ATT", "ATA"]:
            stitched = stitch(seq, codes, sep=sep)
            if "AAAAA" in stitched or "TTTTT" in stitched:
                continue
            out["name"].append(f"{name};{sep}")
            out["seq"].append(stitched)
            out["code1"].append(codes[0])
            out["code2"].append(codes[1])

    return pl.DataFrame(out)


def check_offtargets(dataset: Dataset, constructed: pl.DataFrame, acceptable_tss: list[str]):
    sam = _run_bowtie(dataset, constructed, ignore_revcomp=True)[0]
    df = (
        sam.agg_tm_offtarget(acceptable_tss)
        .filter(
            pl.col("max_tm_offtarget").lt(42)
            & ~pl.col("seq").apply(lambda x: dataset.check_kmers(cast(str, x)))
        )
        .drop("seq")
        .join(constructed, on="name")
        .with_columns(name=pl.col("name").str.split(";").list.first())
        .unique("name")[["name", "code1", "code2", "seq"]]
    )
    return df


# %%
# import json

# %%


# fmt: off
@click.command("construct")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--gene", "-g", type=str)
@click.option("--codebook", "-c", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--target_probes", "-N", type=int, help="Maximum number of probes per gene", default=64)
@click.option("--min_probes", "-n", type=int, help="Minimum number of probes per gene", default=48)
# fmt: on
def _click_construct(
    path: Path,
    output_path: Path,
    gene: str,
    codebook: Path,
    target_probes: int = 64,
    min_probes: int = 48,
):
    res = construct(
        Dataset(path),
        output_path,
        gene=gene,
        codebook=json.loads(codebook.read_text()),
        target_probes=target_probes,
        min_probes=min_probes,
    )
    res.write_parquet(output_path / f"{gene}_final.parquet")


def construct(
    dataset: Dataset,
    output_path: Path | str,
    *,
    gene: str,
    codebook: dict[str, list[int]],
    min_probes: int = 48,
    target_probes: int = 64,
):
    output_path = Path(output_path)
    overlap = assign_overlap(output_path, gene, min_probes=min_probes, target_probes=target_probes)
    screened = pl.read_parquet(
        f"output/{gene}_screened_ol{overlap}.parquet" if overlap else f"output/{gene}_screened.parquet"
    )
    acceptable_tss = pl.read_csv(next(output_path.glob(f"{gene}-*_acceptable_tss.csv")))[
        "transcript_id"
    ].to_list()
    cons = construct_encoding(screened, codebook[gene])

    res = check_offtargets(dataset, cons, acceptable_tss=acceptable_tss).join(
        screened.drop("seq"), on="name", how="left"
    )
    logger.info(f"Constructed {len(res)} probes for {gene}.")
    res.write_parquet(output_path / f"{gene}_final.parquet")
    logger.info(f"Written to {output_path / f'{gene}_final.parquet'}")
    return res


@click.command()
@click.argument("output_path", type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--genes", "-g", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--min-probes", type=int, help="Minimum number of probes per gene", default=48)
@click.option(
    "--out",
    "-o",
    type=click.Path(dir_okay=False, file_okay=True, path_type=Path),
    help="Export filtered gene list",
)
def filter_genes(output_path: Path, genes: Path, min_probes: int, out: Path | None = None):
    def get_probes(gene: str):
        ols = list(output_path.glob(f"{gene}_screened_ol*.parquet"))
        if len(ols) > 0:
            ol = max([int(re.search(r"_ol(\d+)", f.stem).group(1)) for f in ols])
            return pl.read_parquet(output_path / f"{gene}_screened_ol{ol}.parquet")
        return pl.read_parquet(output_path / f"{gene}_screened.parquet")

    gene_list = genes.read_text().splitlines()
    ns: dict[str, int] = {}
    for gene in gene_list:
        if len(screened := get_probes(gene)) < min_probes:
            logger.warning(f"{gene} has {len(screened)} probes, less than {min_probes}.")
        ns[gene] = len(screened)

    if out:
        out.write_text("\n".join(gene for gene, n in ns.items() if n > min_probes))


# readouts = pl.read_csv("data/readout_ref_filtered.csv")
# # genes = Path("zach_26.txt").read_text().splitlines()
# # genes = Path(f"{name}_25.txt").read_text().splitlines()
# smgenes = ["Cdc42", "Neurog2", "Ccnd2"]
# genes, _, _ = gtf_all.check_gene_names(smgenes)
# acceptable_tss = {g: set(pl.read_csv(f"output/{g}_acceptable_tss.csv")["transcript"]) for g in genes}
# n, short_threshold = 67, 65
# # %%
# dfx, overlapped = {}, {}
# for gene in genes:
#     dfx[gene] = GeneFrame.read_parquet(f"output/{gene}_final.parquet")
# dfs = GeneFrame.concat(dfx.values())
# short = dfs.count("gene").filter(pl.col("count") < short_threshold)


# # %%
# fixed_n = {}
# short_fixed = {}


# def run_overlap(genes: Iterable[str], overlap: int):
#     def runpls(gene: str):
#         subprocess.run(
#             ["python", "scripts/new_postprocess.py", gene, "-O", str(overlap)],
#             check=True,
#             capture_output=True,
#         )

#     with ThreadPoolExecutor(32) as executor:
#         for x in as_completed(
#             [
#                 executor.submit(runpls, gene)
#                 for gene in genes
#                 if not Path(f"output/{gene}_final_overlap_{overlap}.parquet").exists()
#             ]
#         ):
#             print("ok")
#             x.result()


#     needs_fixing = set(short["gene"])

#     for ol in [5, 10, 15, 20]:
#         print(ol, needs_fixing)
#         run_overlap(needs_fixing, ol)
#         for gene in needs_fixing.copy():
#             df = GeneFrame.read_parquet(f"output/{gene}_final_overlap_{ol}.parquet")
#             if len(df) >= short_threshold or ol == 20:
#                 needs_fixing.remove(gene)
#                 fixed_n[gene] = ol
#                 short_fixed[gene] = df
#     # else:
#     #     raise ValueError(f"Gene {gene} cannot be fixed")

#     short_fixed = GeneFrame.concat(short_fixed.values())
#     # %%
#     cutted = GeneFrame.concat([dfs.filter(~pl.col("gene").is_in(short["gene"])), short_fixed[dfs.columns]])
#     cutted = GeneFrame(
#         cutted.sort(["gene", "priority"]).groupby("gene").agg(pl.all().head(n)).explode(pl.all().exclude("gene"))
#     )

# %%
