# %%
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from fishtools import hp, rc
from fishtools.mkprobes.starmap.starmap import test_splint_padlock


def run_pipeline(codebook_path: Path, cwd: Path, species: str = "mouse"):
    codebook = json.loads(codebook_path.read_text())
    with ThreadPoolExecutor() as executor:
        for gene in codebook:
            executor.submit(
                subprocess.run,
                f"mkprobes candidates ../data/{species} -o output/ -g {gene}",
                shell=True,
                cwd=cwd,
            )

    with ThreadPoolExecutor() as executor:
        for gene in codebook:
            executor.submit(subprocess.run(f"mkprobes screen output {gene}", shell=True, cwd=cwd))

    with ThreadPoolExecutor() as executor:
        futs = []
        for gene in codebook:
            futs.append(
                executor.submit(
                    subprocess.run,
                    f"mkprobes construct ../data/{species} output -g {gene} -c {codebook_path}",
                    shell=True,
                    cwd=cwd,
                )
            )
    for fut in futs:
        fut.result()


def create_pad(seq: str, readout: str):
    arr = ["GTAGACTA", seq.lower(), readout, "ATTGGTTC"]
    out = "".join(arr)
    if len(out) > 67:
        raise ValueError("Too long")

    arr.insert(2, "TACATAATCAAAT"[: 67 - len(out)])
    return "".join(arr)


def create_spl(seq: str):
    return seq.lower() + "TA GTCTAC GAACCA"


def process(codebook: dict[str, list[int]], cwd: Path, n: int = 16):
    HHP = lambda x: hp(x, "dna")
    dfs = {k: pl.read_parquet(f"{cwd}/output/{k}_final_{bits[0]}.parquet") for k, bits in codebook.items()}
    res = {}
    for df in dfs:
        df_ = (
            dfs[df]
            .with_columns(
                hairpin=pl.max_horizontal(
                    pl.col("cons_splint").map_elements(HHP, return_dtype=pl.Float32),
                    pl.col("cons_pad").map_elements(HHP, return_dtype=pl.Float32),
                ),
                pos_start=pl.col("pos_start").list.get(0),
            )
            .filter(pl.col("pos_start") > 50)
            .filter(pl.col("splint").str.len_chars() > 17)
            .filter(pl.col("padlock").str.len_chars() < 24)
            .sort("hairpin", descending=False)[: n + 2]
            .with_columns(gene=pl.col("name").str.split("_").list.get(0))
            .sort("pos_start")
            .with_row_index("i", offset=0)
        )

        # Filter too close probes.
        for i in reversed(range(len(df_))):
            if df_[i, "pos_start"] - df_[i - 1, "pos_start"] < 50:
                df_ = df_.filter(pl.col("i") != i - 1)

        res[df] = df_.sample(min(n, len(df_)), seed=4)
        print(f"{df}: {len(res[df])}")

    return pl.concat([df.select(next(iter(dfs.values())).columns) for df in res.values()])


def generate_splint_padlock(df: pl.DataFrame, readouts: pl.DataFrame):
    out = []
    for i, row in enumerate(df.iter_rows(named=True)):
        spl = create_spl(rc(row["splint"])).replace(" ", "")
        pad = create_pad(rc(row["padlock"]), readouts[row["code"] - 1, "seq"]).replace(" ", "")
        assert len(spl) < 60, "spl"
        assert len(pad) == 67
        assert test_splint_padlock(spl, pad)
        out.append({"name": f"Spl-{row['gene']}-{i}-R{row['code']}", "seq": spl})
        out.append({"name": f"Pad-{row['gene']}-{i}-R{row['code']}", "seq": pad})

    return pl.DataFrame(out)
