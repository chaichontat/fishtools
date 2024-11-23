# %%
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

path = Path("/home/chaichontat/trc/ficture--leftcortex")

dfs = [
    pl.read_parquet(path.parent / "leftcortex.parquet")
    .with_row_index("idx")
    .filter(pl.col("x").gt(7000) & pl.col("y").lt(-7500)),
    # pl.read_parquet("/mnt/working/e155trcdeconv/registered--leftold/tricycleplus/spots.parquet"),
]


z_ranges = [(i * 5, i * 5 + 10) for i in range(9)]

# %%


def run(path: Path, df: pl.DataFrame, z_range: tuple[int, int] | None = None):
    path = path / f"z{z_range[0]:02d}_{z_range[1]:02d}" if z_range is not None else path
    path.mkdir(exist_ok=True)

    path_ts = path / "transcripts.tsv"
    path_f = path / "features.tsv"
    path_minmax = path / "coordinate_minmax.tsv"

    path_ts.with_suffix(".gz").unlink(missing_ok=True)
    path_f.with_suffix(".gz").unlink(missing_ok=True)
    path_minmax.with_suffix(".gz").unlink(missing_ok=True)

    df = (
        (
            pl.concat(dfs).filter(pl.col("z").is_between(z_range[0], z_range[1]))
            if z_range is not None
            else pl.concat(dfs)
        )
        .select(
            X=pl.col("x") * 0.108,
            Y=pl.col("y") * 0.108,
            gene=pl.col("target").str.split("-").list.get(0),
            MoleculeID=pl.col("idx"),
            Count=pl.lit(1),
        )
        .with_columns(
            X=pl.col("X") - pl.col("X").min() + 1,
            Y=pl.col("Y") - pl.col("Y").min() + 1,
        )
        # .filter(pl.col("X").is_between(300, 900) & pl.col("Y").is_between(-1000, -600))
        .sort("Y")
    )
    df.write_csv(path_ts, separator="\t")
    subprocess.run("pigz -f " + path_ts.as_posix(), shell=True, check=True)

    # X       Y       gene    MoleculeID      Count
    # gene    transcript_id   Count

    gb = (
        df.group_by("gene")
        .len("Count")
        .with_columns(gene=pl.col("gene").str.split("-").list.get(0), transcript_id=pl.col("gene"))
        .select(["gene", "transcript_id", "Count"])
    )

    gb.write_csv(path_f, separator="\t")
    subprocess.run("pigz -f " + path_f.as_posix(), shell=True, check=True)

    with Path(path_minmax).open("w") as f:
        f.write(f"xmin\t{df['X'].min()}\n")
        f.write(f"xmax\t{df['X'].max()}\n")
        f.write(f"ymin\t{df['Y'].min()}\n")
        f.write(f"ymax\t{df['Y'].max()}\n")

    # %%

    batch_size = 100
    batch_buff = 30
    path_out = path / "batched.matrix.tsv.gz"
    path_batch = path / "batched.matrix.tsv"

    subprocess.run(
        f"""ficture run_together \
--in-tsv {path_ts.with_suffix(".tsv.gz")} \
--in-minmax {path_minmax} \
--in-feature {path_f.with_suffix(".tsv.gz")} \
--out-dir {path / "output"} \
--n-jobs 16 \
--gzip "pigz -p 4" \
--train-width 7 \
--plot-each-factor \
--major-axis Y \
--n-factor {i} \
--all""",
        shell=True,
        check=True,
    )

    # %%


with ThreadPoolExecutor(16) as exc:
    futs = []
    for i in range(1):  # z_range in z_ranges:
        futs.append(exc.submit(run, path, dfs[0], z_range=None))

    for f in as_completed(futs):
        f.result()
