# %%
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle, islice
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import polars as pl
import pyfastx
import rich
import rich.rule
import rich.traceback
import rich_click as click
from Bio import Seq
from Bio.Restriction import BamHI, KpnI  # type: ignore
from loguru import logger
from pydantic import BaseModel, TypeAdapter

from fishtools import gen_fasta, hp, rc
from fishtools.mkprobes.codebook.codebook import ProbeSet, hash_codebook
from fishtools.mkprobes.starmap.starmap import generate_head_splint, test_splint_padlock

pl.Config.set_fmt_str_lengths(100)
hfs = pl.read_csv(Path(__file__).parent.parent / "data/headerfooter.csv")
console = rich.get_console()
rich.traceback.install()
species_mapping = {"mouse": "mus musculus", "human": "homo sapiens"}
# %%


# def until_first_g(seq: str, target: str = "G"):
#     r, target = rc(seq.upper()), target.upper()
#     res, truncated = r[:6], r[6:]
#     res += "" if res[-1] == target else truncated[: truncated.index(target) + 1]
#     if len(res) > 12:
#         raise ValueError("No G found")
#     assert res[-1] == target
#     return res


def backfill(seq: str, target: int = 148):
    return (
        "TTCCACTAACTCACATGTCATGCATTATCTTCTATACCTCTGAGCAGATCAGTAGTCTATTACATGCTCGTAGTACCGTAAGCCAGATAC"[
            : max(0, target - len(seq))
        ]
        + seq
    )


def run(path: Path, probeset: ProbeSet, n: int = 16, toolow: int = 4, low: int = 12):
    rand = np.random.default_rng(0)
    idx = probeset.bcidx
    codebook = probeset.load_codebook(path)
    logger.info(f"Loaded {probeset.codebook} with {len(codebook)} genes.")

    tss = list(codebook)
    dfs_ = []
    cols = []

    for ts in tss:
        try:
            df = pl.read_parquet(
                path / f"output/{ts}_final_BamHIKpnI_{','.join(map(str, sorted(codebook[ts])))}.parquet"
            ).sort([
                pl.col("priority").list.max(),
                pl.col("hp").list.min(),
            ])[:n]
        except FileNotFoundError as e:
            logger.critical(e)
            continue

        if len(df) < low:
            # Resample to prevent probe dropout. Capping at 2x coverage.
            df = pl.concat([df, df[: min(low - len(df), len(df))]])

        if not cols:
            cols = df.columns

        if df["gene"].dtype == pl.List:
            df = df.with_columns(gene=pl.col("gene").list.get(0))

        dfs_.append(df[:, cols])

    # if len(dfs_) != len(tss):
    #     raise Exception("Gene count mismatch")

    dfs = pl.concat(dfs_)

    outpath = Path(path / "generated" / probeset.name)
    outpath.mkdir(exist_ok=True, parents=True)
    # RepeatMasker
    if probeset.species not in species_mapping:
        probeset.species = "mouse"

    with ThreadPoolExecutor() as exc:
        for col_name in ["splint", "padlock"]:
            (outpath / f"{col_name}.fasta").write_text(gen_fasta(dfs[col_name]).getvalue())
            exc.submit(
                subprocess.run,
                f'RepeatMasker -pa 16 -norna -s -no_is -species "{species_mapping[probeset.species]}" {outpath / f"{col_name}.fasta"}',
                shell=True,
                check=True,
            )

    dfs = dfs.with_columns({
        col_name: [seq for name, seq in pyfastx.Fastx((outpath / f"{col_name}.fasta.masked").as_posix())]
        for col_name in ["splint", "padlock"]
        if (outpath / f"{col_name}.fasta.masked").exists()
    }).filter(~pl.col("splint").str.contains("N") & ~pl.col("padlock").str.contains("N"))

    counts = dfs.group_by("gene").len(name="count")
    # Since we resample low counting probes, we need to double the threshold.
    if (bad := counts.filter(pl.col("count") < toolow * 2)).__len__():
        print(bad)
        raise ValueError("Not enough probes")

    # Before
    spl_idx = idx * 2
    pad_idx = idx * 2 + 1

    def padpad(s: str, target: int = 99):
        if len(s) > target + 2:
            raise ValueError("Too long")
        if len(s) > target:
            return s
        return s + "AATCACATAAAT"[: target - len(s)]

    # This is padlock.
    logger.info("Generating splint header.")
    res = dfs.with_columns(
        pad_cut=(
            # head
            hfs[pad_idx, "header"][-3:]
            + pl.col("padlock")
            .map_elements(lambda x: generate_head_splint(x, rand), return_dtype=pl.Utf8)
            .str.to_lowercase()
            + "ta"  # what the paper uses
            + pl.col("seq").map_elements(rc, return_dtype=pl.Utf8)
        ).map_elements(padpad, return_dtype=pl.Utf8)
        + "at"
        + rc(hfs[spl_idx, "footer"][:3])
        + hfs[pad_idx, "footer"][:3]
    )

    it = cycle("ATAAT")

    def splint_pad(seq: str, target: int = 47):
        if len(seq) > target:
            return seq
        return "".join(islice(it, target - len(seq))) + seq

    # Splint
    res = (
        res.with_columns(
            spl_cut=(
                "TGTTGATGAGGTGTTGATGAATA"
                + pl.col("splint").map_elements(rc, return_dtype=pl.Utf8)
                + "ca"
                + pl.col("pad_cut").str.slice(0, 6).map_elements(rc, return_dtype=pl.Utf8)
                + pl.col("pad_cut").str.slice(-6, 6).map_elements(rc, return_dtype=pl.Utf8)
            ).map_elements(splint_pad, return_dtype=pl.Utf8)
        )
    ).filter(
        (
            pl.col("spl_cut")
            .map_elements(lambda x: BamHI.search(Seq.Seq(x)), return_dtype=pl.List(pl.UInt32))
            .list.len()
            == 0
        )
        & (
            pl.col("spl_cut")
            .map_elements(lambda x: KpnI.search(Seq.Seq(x)), return_dtype=pl.List(pl.UInt32))
            .list.len()
            == 0
        )
        & (
            pl.col("pad_cut")
            .map_elements(lambda x: BamHI.search(Seq.Seq(x)), return_dtype=pl.List(pl.UInt32))
            .list.len()
            == 0
        )
        & (
            pl.col("pad_cut")
            .map_elements(lambda x: KpnI.search(Seq.Seq(x)), return_dtype=pl.List(pl.UInt32))
            .list.len()
            == 0
        )
    )

    def double_digest(s: str) -> str:
        return BamHI.catalyze(KpnI.catalyze(Seq.Seq(s))[1])[0].__str__()

    for s, r in zip(res["spl_cut"], res["pad_cut"]):
        assert test_splint_padlock(s, r, lengths=(6, 6)), (s, r)

    out: pl.DataFrame = res.with_columns(
        # restriction scar already accounted for
        splintcons=hfs[spl_idx, "header"] + pl.col("spl_cut") + hfs[spl_idx, "footer"][3:],
        padlockcons=hfs[pad_idx, "header"][:-3].lower() + pl.col("pad_cut") + hfs[pad_idx, "footer"][3:],
    ).with_columns(splintcons=pl.col("splintcons").map_elements(backfill, return_dtype=pl.Utf8))

    for s, r in zip(out["splintcons"], out["padlockcons"]):
        assert test_splint_padlock(*map(double_digest, (s, r)), lengths=(6, 6)), (s, r)

    assert (out["padlockcons"].str.len_chars().is_between(139, 150)).all()

    (gen_path := path / "generated").mkdir(exist_ok=True, parents=True)
    out.write_parquet(gen_path / (probeset.name + ".parquet"))
    logger.info(f"{len(out)} probe pairs written to {gen_path / (probeset.name + '.parquet')}")

    (gen_path / (probeset.name + "_pad.fasta")).write_text(
        gen_fasta(out["padlockcons"], names=range(len(out))).getvalue()
    )
    (gen_path / (probeset.name + "_splint.fasta")).write_text(
        gen_fasta(out["splintcons"], names=range(len(out))).getvalue()
    )
    return out


# %%
@click.group()
@click.argument("manifest", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def cli(ctx: click.Context, manifest: Path):
    ctx.ensure_object(dict)
    mfs = TypeAdapter(list[ProbeSet]).validate_json(Path(manifest).read_text())
    ctx.obj["manifest"] = mfs
    ctx.obj["path"] = manifest.parent


@cli.command()
@click.argument("short", type=int)
@click.option("--verbose", "-v", is_flag=True)
@click.option(
    "--delete",
    is_flag=True,
    help="Delete the probes from a .tss.txt file that are too short.",
)
@click.option(
    "--permanent",
    is_flag=True,
    help="Delete the probes from a .tss.txt file that are too short permanently.",
)
@click.pass_context
def short(
    ctx: click.Context, short: int, verbose: bool = False, delete: bool = False, permanent: bool = False
):
    """
    Identifies and optionally removes transcripts with fewer probes than a specified threshold.

    This function iterates through probe sets defined in the manifest. For each probe set,
    it loads the corresponding codebook and associated parquet files containing probe data.
    It then counts the number of probes per gene.

    If the '--delete' flag is set, genes with probe counts below the 'short' threshold
    are removed from a copy of the .tss.txt file.
    If '--permanent' is also set, the original .tss.txt file is overwritten.
    Otherwise, a new file with the suffix '.tss.ok.txt' is created.

    Args:
        ctx: The Click context, containing the manifest and path.
        short: The minimum number of probes a gene must have.
        verbose: If True, prints detailed information about genes with too few probes.
        delete: If True, removes genes with too few probes from the .tss.txt file.
        permanent: If True (and 'delete' is True), overwrites the original .tss.txt file.
                   Otherwise, a new file with '.tss.ok.txt' suffix is created.

    Raises:
        ValueError: If '--permanent' is used without '--delete'.
        ValueError: If genes marked for deletion are not found in the .tss.txt file.
    """

    if permanent and not delete:
        raise ValueError("Cannot use --permanent without --delete")

    manifest: list[ProbeSet] = ctx.obj["manifest"]
    path_main: Path = ctx.obj["path"]

    COL_NAME = "gene"
    for probeset in manifest:
        console.print(rich.rule.Rule(title=probeset.name, align="left"))
        baddies = []

        codebook = probeset.load_codebook(path_main)

        path = (path_main / probeset.codebook).parent

        tss = list(codebook)
        dfs_ = []
        for ts in tss:
            try:
                _df = pl.read_parquet(
                    path / f"output/{ts}_final_BamHIKpnI_{','.join(map(str, sorted(codebook[ts])))}.parquet"
                )
                _df = _df.sort([pl.col("priority").list.min(), pl.col("hp").list.max()])
                dfs_.append(
                    _df.select([
                        "name",
                        "seq",
                        "code1",
                        "code2",
                        "code3",
                        "index",
                        "id",
                        "flag",
                        "transcript",
                        "pos",
                        "cigar",
                        "aln_score",
                        "aln_score_best",
                        "n_ambiguous",
                        "n_mismatches",
                        "n_opens",
                        "n_extensions",
                        "edit_distance",
                        "mismatched_reference",
                        "gene",
                        "transcript_ori",
                        "pos_start",
                        "pos_end",
                        "length",
                        "match",
                        "match_consec",
                        "pad_start",
                        "maps_to_pseudo",
                        "max_tm_offtarget",
                        "match_consec_all",
                        "ok_quad_c",
                        "ok_quad_a",
                        "ok_stack_c",
                        "ok_comp_a",
                        "gc_content",
                        "ok_gc",
                        "tm",
                        "hp",
                        "oks",
                        "priority",
                        "splint",
                        "padlock",
                        "seqori",
                    ])
                )
            # .sample(shuffle=True, seed=4, fraction=1)

            except FileNotFoundError:
                baddies.append(probeset.name)
                # if verbose:
                logger.warning(
                    "File "
                    + f"output/{ts}_final_BamHIKpnI_{','.join(map(str, sorted(codebook[ts])))}.parquet"
                    + " not found."
                )

        if not len(dfs_):
            logger.warning(f"No data for {probeset.name} at {probeset.codebook}")
            continue

        dfs: pl.DataFrame = pl.concat(dfs_)

        counts = dfs.group_by(COL_NAME).len(name="count")
        if len(bad := counts.filter(pl.col("count") < short)):
            # print(probeset.name)
            baddies = bad[COL_NAME].to_list()
            if verbose:
                rich.print(bad)
            else:
                rich.print("\n".join(bad[COL_NAME].to_list()))
            print(f"Found {len(bad)} genes with fewer than {short} probes.")
        else:
            logger.info(f"All genes have at least {short} probes.")

        if delete:
            tss = probeset.load_codebook(path)
            # Check
            genes = {("-".join(ts.split("-")[:-1]) if "-" in ts else ts) for ts in tss}
            baddies = set(baddies)
            if not baddies.issubset(genes):
                raise ValueError(f"{baddies - genes} not found in .tss.txt file. Wrong file?")

            goodies = [ts for ts in tss if ("-".join(ts.split("-")[:-1]) if "-" in ts else ts) not in baddies]
            logger.info(
                f"Deleted {len(tss) - len(goodies)} genes from {Path(probeset.codebook).with_suffix('.tss.txt')}"
            )
            out_path = (
                path / probeset.codebook
                if permanent
                else (path / probeset.codebook).with_suffix(".tss.ok.txt")
            )
            out_path.with_suffix(".tss.txt").write_text("\n".join(sorted(goodies)))


@cli.command()
@click.pass_context
def gen(ctx: click.Context):
    mfs: list[ProbeSet] = ctx.obj["manifest"]
    path: Path = ctx.obj["path"]

    total_probes = 0
    for x in mfs:
        if isinstance(x.n_probes, int):
            n = x.n_probes
            low = min(n, 13)
        elif x.n_probes is not None:
            n = 34 if x.n_probes == "high" else 16
            low = 24 if x.n_probes == "high" else 12
        else:
            n = 34 if x.species == "human" else 16
            low = 24 if x.species == "human" else 12

        total_probes += len(run(path, x, n=n, toolow=4, low=low)) * 2
        logger.info(f"Cumulative probes: {total_probes}")

    # with ThreadPoolExecutor(4) as exc:
    #     for m in mfs:
    #         for ps in ["splint"]:
    #             exc.submit(
    #                 subprocess.run,
    #                 f'RepeatMasker -pa 16 -norna -s -no_is -species "{species_mapping[m.species]}" {path}/generated/{m.name}_{ps}.fasta',
    #                 shell=True,
    #                 check=True,
    #             )

    superout = []
    for m in mfs:
        out = []
        paths = [
            (path / "generated" / f"{m.name}_splint.fasta"),
            (path / "generated" / f"{m.name}_pad.fasta"),
        ]

        for i in range(2):
            if not paths[i].exists():
                raise FileNotFoundError(f"File {paths[i]} not found.")
                # paths[i] = paths[i].with_name(paths[i].name[:-7])

        for s, p in zip(*[pyfastx.Fastx(p.as_posix()) for p in paths]):
            if "N" not in s[1] and "N" not in p[1]:
                out.append(s[1])
                out.append(p[1])
        Path(path / "generated" / f"{m.name}_final.txt").write_text("\n".join(out))
        superout.extend(out)

    Path(path / "generated" / f"_allout{int(time.time())}.txt").write_text("\n".join(superout))


if __name__ == "__main__":
    cli()


# t7 = "TAATACGACTCACTATAGGG"
# assert out["padlockcons"].str.contains(rc(t7)[:5]).all()

# %%


# for name in ["genestarpad.fasta", "genestarsplint.fasta"]:

# cons = dfs.with_columns(constructed=header + pl.col("seq") + footer)
# cons = cons.with_columns(constructed=pl.col("constructed").map_elements(backfill))
# # %%
# import pyfastx


# # %%

# Path("starwork/genestar_out.txt").write_text("\n".join(out))


# %%

# %%
