# %%
import json
import re
from itertools import chain, cycle, permutations
from pathlib import Path
from typing import Callable, Collection, Final, Iterable, Sequence, cast

import click
import polars as pl
from loguru import logger

from fishtools import rc
from fishtools.mkprobes.candidates import _run_bowtie
from fishtools.mkprobes.ext.external_data import Dataset
from fishtools.mkprobes.starmap.starmap import split_probe

READOUTS: Final[dict[int, str]] = {
    x["id"]: x["seq"] for x in pl.read_csv(Path(__file__).parent / "readout_ref_filtered.csv").to_dicts()
}


def assign_overlap(
    output: Path | str,
    gene: str,
    *,
    target_probes: int = 16,
    max_overlap: int = 5,
    restriction: str = "BsaI",
) -> int:
    if max_overlap % 5 != 0 or max_overlap < 0:
        raise ValueError("max_overlap must be a multiple of 5")
    output = Path(output)
    for ol in chain((-2,), range(5, max_overlap + 1, 5)):
        df = pl.read_parquet(output / f"{gene}_screened_ol{ol}{restriction}.parquet")
        if len(df) >= target_probes:
            return ol

    # if len(df) >= min_probes:  # type: ignore
    return ol  # type: ignore

    # raise ValueError(f"Gene {gene} cannot be fixed")


def stitch(seq: str, codes: Sequence[int], sep: str = "TT") -> str:
    return sep.join(rc(READOUTS[c]) for c in codes[:-1]) + seq + rc(READOUTS[codes[-1]])


def construct_encoding(seq_encoding: pl.DataFrame, idxs: list[int], n: int = -1):
    if n == -1:
        n = len(idxs)
    if len(idxs) != n:
        raise ValueError(f"Invalid number of readouts: {idxs}")
    if any(idx not in READOUTS for idx in idxs):
        raise ValueError(f"Invalid readout indices: {idxs}")

    pairs = cast(Iterable[tuple[int, int]], cycle(permutations(idxs, n)))

    out = dict(name=[], seq=[], spacer=[])
    for i in range(n):
        out[f"code{i+1}"] = []

    for name, pad, padstart in seq_encoding[["name", "padlock", "padstart"]].iter_rows():
        for codes, _ in zip(pairs, range(4)):
            assert padstart > 17
            for sep in ["TA", "AA", "AT", "AA"]:
                stitched = stitch(pad, codes, sep=sep)
                if "AAAAA" in stitched or "TTTTT" in stitched or "CCCCC" in stitched or "GGGGG" in stitched:
                    continue
                out["name"].append(f"{name};;{sep}{','.join(map(str,codes))}")
                out["spacer"].append("")
                out["seq"].append(stitched)
                for i, code in enumerate(codes):
                    out[f"code{i+1}"].append(code)

    return pl.DataFrame(out)


def check_offtargets(dataset: Dataset, constructed: pl.DataFrame, acceptable_tss: list[str]):
    sam = _run_bowtie(dataset, constructed, ignore_revcomp=True)[0]
    acc = sam.agg_tm_offtarget(acceptable_tss)
    df = (
        acc.filter(
            pl.col("max_tm_offtarget").lt(37)
            & ~pl.col("seq").apply(lambda x: dataset.check_kmers(cast(str, x)))
        )
        .drop("seq")
        .join(constructed, on="name", suffix="padlock", how="inner")
        .with_columns(name=pl.col("name").str.split(";;").list.first())
        .sort("match", descending=True)
        .unique("name", keep="first", maintain_order=True)
    )
    return df


# %%


# fmt: off
@click.command("construct")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--gene", "-g", type=str)
@click.option("--codebook", "-c", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--target_probes", "-N", type=int, help="Maximum number of probes per gene", default=72)
@click.option("--restriction", multiple=True, type=str, help="Restriction enzymes to use")
# fmt: on
def _click_construct(
    path: Path,
    output_path: Path,
    gene: str,
    codebook: Path,
    target_probes: int = 72,
    restriction: list[str] | str | None = None,
):
    construct(
        Dataset(path),
        output_path,
        transcript=gene,
        codebook=json.loads(codebook.read_text()),
        target_probes=target_probes,
        restriction=restriction,
    )


def construct(
    dataset: Dataset,
    output_path: Path | str,
    *,
    transcript: str,
    codebook: dict[str, list[int]],
    target_probes: int = 64,
    restriction: list[str] | str | None = None,
    construction_function: Callable | None = None,
):
    output_path = Path(output_path)
    if isinstance(restriction, Collection):
        restriction = "_" + "".join(restriction)
    restriction = restriction or ""

    # if (final_path := Path(output_path / f"{transcript}_final{restriction}.parquet")).exists() and set(
    #     pl.read_parquet(final_path)[["code1", "code2"]].to_numpy().flatten()
    # ) != set(codebook[transcript]):
    #     logger.critical(f"Codebook for {transcript} has changed.")
    # exit(1)

    overlap = assign_overlap(output_path, transcript, target_probes=target_probes, restriction=restriction)
    screened = pl.read_parquet(
        scr_path := output_path / f"{transcript}_screened_ol{overlap}{restriction}.parquet"
    )
    logger.debug(f"Using {scr_path} for {transcript}.")

    assert not screened["seq"].str.contains("N").any()
    acceptable_tss = pl.read_csv(next(output_path.glob(f"{transcript}_acceptable_tss.csv")))[
        "transcript_id"
    ].to_list()

    screened = (
        screened.with_columns(splitted=pl.col("seq").apply(lambda x: split_probe(rc(x), 58)))
        .with_columns(
            splint=pl.col("splitted").list.get(0).apply(rc, return_dtype=pl.Utf8),
            padlock=pl.col("splitted").list.get(1).apply(rc, return_dtype=pl.Utf8),
            padstart=pl.col("splitted").list.get(2).cast(pl.Int16),
        )
        .drop("splitted")
        .filter(pl.col("padstart") > 0)
    )

    cons = construct_encoding(screened, codebook[transcript])
    res = check_offtargets(dataset, cons, acceptable_tss=acceptable_tss).join(
        screened.rename(dict(seq="seqori")), on="name", how="left"
    )

    logger.info(f"Constructed {len(res)} probes for {transcript}.")
    assert res["seq"].is_not_null().all()
    res.write_parquet(output_path / f"{transcript}_final{restriction}.parquet")
    logger.info(f"Written to {output_path / f'{transcript}_final{restriction}.parquet'}")
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
