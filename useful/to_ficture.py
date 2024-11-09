# %%
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

path = Path("/mnt/working/lai/ficture")
path_ts = path / "transcripts.tsv"
path_f = path / "features.tsv"
path_minmax = path / "coordinate_minmax.tsv"
path.mkdir(exist_ok=True)

dfs = [
    pl.read_parquet(path.parent / "registered--whole_embryo/genestar/spots.parquet"),
    # pl.read_parquet("/mnt/working/e155trcdeconv/registered--leftold/tricycleplus/spots.parquet"),
]


df = (
    pl.concat(dfs)
    .filter(pl.col("z") > 1)
    .select(
        X=pl.col("x") * 0.216,
        Y=pl.col("y") * 0.216,
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
print(len(df))
subprocess.run("pigz -f " + path_ts.as_posix(), shell=True, check=True)
# %%
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
# %%
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

# subprocess.run(
#     f"""ficture make_spatial_minibatch \
#         --input {path_ts.with_suffix(".tsv.gz")} \
#         --output {path_batch} \
#         --mu_scale {1 / 0.108} \
#         --batch_size ${batch_size} \
#         --batch_buff ${batch_buff} \
#         --major_axis Y""",
#     shell=True,
#     check=True,
# )

# %%
with ThreadPoolExecutor(16) as exc:
    for i in [24, 30]:
        exc.submit(
            subprocess.run,
            f"""ficture run_together \
        --in-tsv {path_ts.with_suffix(".tsv.gz")} \
        --in-minmax {path_minmax} \
        --in-feature {path_f.with_suffix(".tsv.gz")} \
        --out-dir {path / "output"} \
        --n-jobs 16 \
        --gzip "pigz -p 4" \
        --train-width 10 \
        --plot-each-factor \
        --major-axis Y \
        --n-factor {i} \
        --all""",
            shell=True,
            check=True,
        )

# %%
