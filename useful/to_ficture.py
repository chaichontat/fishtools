# %%
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from shapely.geometry import MultiPolygon, Polygon

from fishtools.segment.overlay_spots import (
    assign_spots_to_polygons,
    build_spatial_index,
    extract_polygons_from_roifile,
)
from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.utils.utils import noglobal

roi = "big"
path = Path(f"/working/20250612_ebe00219_3/analysis/deconv/ficture--{roi}")
path.mkdir(exist_ok=True, parents=True)
dfs = [
    pl.scan_parquet(path.parent.as_posix() + f"/{roi}*.parquet").collect().with_row_index("idx")
    # .filter(pl.col("y").is_between(-7500, 0)),
    # pl.read_parquet("/mnt/working/e155trcdeconv/registered--leftold/tricycleplus/spots.parquet"),
]


tc = TileConfiguration.from_file(path.parent / f"stitch--{roi}" / "TileConfiguration.registered.txt")
coords = tc.df
x_offset = coords["x"].min()
y_offset = coords["y"].min()

rois = extract_polygons_from_roifile(path.parent / f"stitch--{roi}+edu/RoiSet.zip", 0, 0.5)

tree, idxs = build_spatial_index(rois, 0)
df = (
    pl.concat(dfs)
    .with_columns(x_adj=pl.col("x") - x_offset, y_adj=pl.col("y") - y_offset)
    .with_row_index("spot_id")
)


# %%
path_subsetted = path / "whitebody.parquet"
if not path_subsetted.exists():
    assigned = assign_spots_to_polygons(df, tree, idxs, rois, 0)
    subsetted = assigned.join(df, "spot_id")
    subsetted.write_parquet(path_subsetted)
else:
    subsetted = pl.read_parquet(path / "whitebody.parquet")

subsetted = pl.scan_parquet(f"{path.parent}/*/whitebody.parquet").collect()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)


for poly, meta in rois:
    if poly.is_empty:
        continue

    def plot_poly_outline(p, ax):
        x, y = p.exterior.xy
        ax.plot(np.array(x), np.array(y), color="cyan", linewidth=0.7)

    if isinstance(poly, Polygon):
        plot_poly_outline(poly, ax)
    elif isinstance(poly, MultiPolygon):
        for p_geom in poly.geoms:
            if isinstance(p_geom, Polygon):
                plot_poly_outline(p_geom, ax)


fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(subsetted["x"][::10], subsetted["y"][::10], s=0.1, alpha=0.5)
# ax.set_xlim(0, 18000)
# ax.set_ylim(0, 18000)
ax.set_aspect("equal")
ax.invert_yaxis()
# subsetted.write_parquet(path / "whitebody.parquet")
# %%


@noglobal
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
        (df.filter(pl.col("z").is_between(z_range[0], z_range[1])) if z_range is not None else pl.concat(dfs))
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
--train-width 6 \
--plot-each-factor \
--major-axis Y \
--n-factor 25 \
--all""",
        shell=True,
        check=True,
    )


# %%


with ThreadPoolExecutor(16) as exc:
    futs = []
    for i in range(1):  # z_range in z_ranges:
        futs.append(exc.submit(run, path, subsetted, z_range=(10, 30)))

    for f in as_completed(futs):
        f.result()

# %%
