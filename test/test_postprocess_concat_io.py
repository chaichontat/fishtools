"""Tests for fishtools.postprocess.io_concat data loaders and assemblers."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from loguru import logger

from fishtools.postprocess.io_concat import (
    ConcatDataError,
    ConcatDataSpec,
    arrange_rois,
    compute_weighted_centroids,
    join_ident_with_spots,
    load_ident_tables,
    load_intensity_tables,
    load_polygon_tables,
    load_spot_tables,
    merge_polygons_with_intensity,
)


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _workspace_root(tmp_path: Path) -> Path:
    (tmp_path / "analysis" / "deconv").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def concat_spec(tmp_path: Path) -> ConcatDataSpec:
    root = _workspace_root(tmp_path)
    return ConcatDataSpec(
        workspace=root,
        rois=("roi1",),
        seg_codebook="seg",
        analysis_codebooks=("cb1",),
    )


def test_load_ident_tables_success(tmp_path: Path, concat_spec: ConcatDataSpec) -> None:
    ident_dir = (
        tmp_path
        / "analysis"
        / "deconv"
        / "stitch--roi1+seg"
        / "chunks+cb1"
    )
    _write_parquet(
        pl.DataFrame({
            "spot_id": [1, 2],
            "label": [100, 101],
            "target": ["Gene1-20", "Gene2-21"],
        }),
        ident_dir / "ident_000.parquet",
    )

    tables = load_ident_tables(concat_spec)

    key = ("roi1", "cb1")
    assert key in tables
    df = tables[key]
    assert df.height == 2
    assert df["roi"].unique().to_list() == ["roi1"]
    assert df["codebook"].unique().to_list() == ["cb1"]
    assert df["roilabel"].to_list() == ["roi1100", "roi1101"]
    assert df["z"].to_list() == [0, 0]


def test_load_ident_tables_missing(tmp_path: Path, concat_spec: ConcatDataSpec) -> None:
    with pytest.raises(ConcatDataError, match="No identification files"):
        load_ident_tables(concat_spec)


def test_load_intensity_tables_success(tmp_path: Path, concat_spec: ConcatDataSpec) -> None:
    intensity_dir = (
        tmp_path
        / "analysis"
        / "deconv"
        / "stitch--roi1+seg"
        / "intensity_cfse"
    )
    _write_parquet(
        pl.DataFrame({
            "label": [100, 101],
            "mean_intensity": [10.0, 11.0],
        }),
        intensity_dir / "intensity-000.parquet",
    )

    tables = load_intensity_tables(concat_spec, channel="cfse")
    df = tables["roi1"]
    assert df.height == 2
    assert set(df.columns) >= {"roi", "roilabel", "roilabelz"}
    assert df["roi"].unique().to_list() == ["roi1"]


def test_load_intensity_tables_optional(tmp_path: Path, concat_spec: ConcatDataSpec) -> None:
    tables = load_intensity_tables(concat_spec, channel="cfse", required=False)
    assert tables == {}


def test_load_spot_tables_success(tmp_path: Path, concat_spec: ConcatDataSpec) -> None:
    spots_path = tmp_path / "analysis" / "deconv" / "roi1+cb1.parquet"
    _write_parquet(
        pl.DataFrame({
            "index": [100, 101],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
        }),
        spots_path,
    )

    tables = load_spot_tables(concat_spec)
    df = tables[("roi1", "cb1")]
    assert set(df.columns) >= {"label", "roi", "codebook"}
    assert df["label"].to_list() == [100, 101]


def test_load_spot_tables_missing(tmp_path: Path, concat_spec: ConcatDataSpec) -> None:
    with pytest.raises(ConcatDataError, match="No spot parquet"):
        load_spot_tables(concat_spec)


def test_load_polygon_tables_success(tmp_path: Path, concat_spec: ConcatDataSpec) -> None:
    polygon_dir = (
        tmp_path
        / "analysis"
        / "deconv"
        / "stitch--roi1+seg"
        / "chunks+cb1"
    )
    _write_parquet(
        pl.DataFrame({
            "label": [100, 101],
            "area": [50.0, 75.0],
            "centroid_x": [10.0, 20.0],
            "centroid_y": [30.0, 40.0],
        }),
        polygon_dir / "polygons_000.parquet",
    )

    tables = load_polygon_tables(concat_spec)
    df = tables["roi1"]
    assert set(df.columns) >= {"roilabel", "roilabelz", "roi", "area"}
    assert df["roilabel"].to_list() == ["roi1100", "roi1101"]


def test_join_ident_with_spots_success() -> None:
    ident_tables = {
        ("roi1", "cb1"): pl.DataFrame({
            "label": [100, 101],
            "codebook": ["cb1", "cb1"],
            "roi": ["roi1", "roi1"],
            "spot_id": [1, 2],
        })
    }
    spot_tables = {
        ("roi1", "cb1"): pl.DataFrame({
            "label": [100, 101],
            "codebook": ["cb1", "cb1"],
            "x": [1.5, 2.5],
            "y": [3.5, 4.5],
        })
    }

    joined = join_ident_with_spots(ident_tables, spot_tables)
    assert joined.height == 2
    assert set(["x", "y"]).issubset(joined.columns)


def test_merge_polygons_with_intensity_success() -> None:
    polygons = {
        "roi1": pl.DataFrame({
            "roilabelz": ["0roi1100", "0roi1101"],
            "roilabel": ["roi1100", "roi1101"],
            "roi": ["roi1", "roi1"],
            "area": [10.0, 20.0],
            "centroid_x": [1.0, 2.0],
            "centroid_y": [3.0, 4.0],
            "z": [0, 0],
        })
    }
    intensities = {
        "roi1": pl.DataFrame({
            "roilabelz": ["0roi1100", "0roi1101"],
            "mean_intensity": [5.0, 6.0],
            "max_intensity": [7.0, 8.0],
            "min_intensity": [2.0, 3.0],
        })
    }

    merged = merge_polygons_with_intensity(polygons, intensities)
    assert merged.height == 2
    assert set(["mean_intensity", "max_intensity", "min_intensity"]).issubset(merged.columns)


def test_arrange_rois() -> None:
    polygons = pl.DataFrame({
        "roi": ["a", "a", "b"],
        "centroid_x": [0.0, 10.0, 5.0],
        "centroid_y": [0.0, 10.0, 5.0],
        "area": [1.0, 1.0, 1.0],
        "roilabel": ["a1", "a2", "b1"],
        "z": [0, 0, 0],
    })

    arranged, offsets = arrange_rois(polygons, max_columns=1, padding=10.0)

    assert set(offsets.keys()) == {"a", "b"}
    assert arranged.filter(pl.col("roi") == "b")["centroid_y"].min() >= 10.0


def test_compute_weighted_centroids() -> None:
    polygons = pl.DataFrame({
        "roilabel": ["roi1100", "roi1100", "roi1101"],
        "roi": ["roi1", "roi1", "roi1"],
        "area": [10.0, 20.0, 30.0],
        "centroid_x": [0.0, 5.0, 10.0],
        "centroid_y": [0.0, 5.0, 10.0],
        "z": [0, 1, 0],
        "mean_intensity": [1.0, 3.0, 5.0],
        "max_intensity": [2.0, 4.0, 6.0],
        "min_intensity": [0.5, 1.0, 1.5],
    })

    centroids = compute_weighted_centroids(polygons)
    row = centroids.filter(pl.col("roilabel") == "roi1100").row(0, named=True)
    assert pytest.approx(row["x"]) == (0.0 * 10 + 5.0 * 20) / 30
    assert pytest.approx(row["mean_intensity"]) == (1.0 * 10 + 3.0 * 20) / 30


def test_join_ident_with_spots_warns_when_missing(caplog: pytest.LogCaptureFixture) -> None:
    ident_tables = {
        ("roi1", "cb1"): pl.DataFrame({
            "label": [1],
            "codebook": ["cb1"],
            "roi": ["roi1"],
        }),
        ("roi2", "cb1"): pl.DataFrame({
            "label": [2],
            "codebook": ["cb1"],
            "roi": ["roi2"],
        }),
    }

    spot_tables = {
        ("roi1", "cb1"): pl.DataFrame({
            "label": [1],
            "codebook": ["cb1"],
        })
    }

    handler_id = logger.add(caplog.handler, level="WARNING")
    try:
        joined = join_ident_with_spots(ident_tables, spot_tables)
    finally:
        logger.remove(handler_id)

    assert joined.height == 1
    assert "Missing spot join data" in caplog.text


def test_merge_polygons_with_intensity_warns(caplog: pytest.LogCaptureFixture) -> None:
    polygons = {
        "roi1": pl.DataFrame({
            "roilabelz": ["0roi1100"],
            "roilabel": ["roi1100"],
            "roi": ["roi1"],
            "area": [1.0],
            "centroid_x": [1.0],
            "centroid_y": [1.0],
            "z": [0],
        }),
        "roi2": pl.DataFrame({
            "roilabelz": ["0roi2100"],
            "roilabel": ["roi2100"],
            "roi": ["roi2"],
            "area": [1.0],
            "centroid_x": [1.0],
            "centroid_y": [1.0],
            "z": [0],
        }),
    }

    intensities = {
        "roi1": pl.DataFrame({
            "roilabelz": ["0roi1100"],
            "mean_intensity": [5.0],
        })
    }

    handler_id = logger.add(caplog.handler, level="WARNING")
    try:
        merged = merge_polygons_with_intensity(polygons, intensities)
    finally:
        logger.remove(handler_id)

    assert merged.filter(pl.col("roi") == "roi1").height == 1
    assert "Missing intensity tables" in caplog.text


def test_arrange_rois_multi_column() -> None:
    polygons = pl.DataFrame({
        "roi": ["a", "b", "c"],
        "centroid_x": [0.0, 5.0, 2.5],
        "centroid_y": [0.0, 5.0, 1.0],
        "area": [1.0, 1.0, 1.0],
        "roilabel": ["a0", "b0", "c0"],
        "z": [0, 0, 0],
    })

    arranged, offsets = arrange_rois(polygons, max_columns=2, padding=2.0)

    assert set(offsets) == {"a", "b", "c"}
    # ROI c should be on the second row (grid row 1) due to max_columns=2
    assert arranged.filter(pl.col("roi") == "c")["centroid_y"].min() > arranged["centroid_y"].min()


def test_compute_weighted_centroids_handles_missing_intensity() -> None:
    polygons = pl.DataFrame({
        "roilabel": ["roi1100", "roi1101"],
        "roi": ["roi1", "roi1"],
        "area": [5.0, 5.0],
        "centroid_x": [0.0, 10.0],
        "centroid_y": [0.0, 10.0],
        "z": [0, 0],
    })

    centroids = compute_weighted_centroids(polygons)
    assert centroids.height == 2
    assert "mean_intensity" not in centroids.columns


def test_concat_end_to_end(tmp_path: Path) -> None:
    root = _workspace_root(tmp_path)
    spec = ConcatDataSpec(
        workspace=root,
        rois=("roi-1", "roi-2"),
        seg_codebook="seg",
        analysis_codebooks=("cb1",),
    )

    for roi in spec.rois:
        chunks_dir = (
            root
            / "analysis"
            / "deconv"
            / f"stitch--{roi}+seg"
            / "chunks+cb1"
        )
        _write_parquet(
            pl.DataFrame({
                "spot_id": [1, 2],
                "label": [100, 101],
                "target": ["GeneA-10", "GeneB-11"],
            }),
            chunks_dir / "ident_000.parquet",
        )
        _write_parquet(
            pl.DataFrame({
                "label": [100, 101],
                "area": [20.0, 30.0],
                "centroid_x": [5.0, 15.0],
                "centroid_y": [2.0, 8.0],
            }),
            chunks_dir / "polygons_000.parquet",
        )

        intensity_dir = (
            root
            / "analysis"
            / "deconv"
            / f"stitch--{roi}+seg"
            / "intensity_cfse"
        )
        _write_parquet(
            pl.DataFrame({
                "label": [100, 101],
                "mean_intensity": [100.0, 200.0],
                "max_intensity": [150.0, 220.0],
                "min_intensity": [80.0, 190.0],
            }),
            intensity_dir / "intensity-000.parquet",
        )

        spots_path = root / "analysis" / "deconv" / f"{roi}+cb1.parquet"
        _write_parquet(
            pl.DataFrame({
                "index": [100, 101],
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
            }),
            spots_path,
        )

    ident = load_ident_tables(spec)
    spots = load_spot_tables(spec)
    polygons = load_polygon_tables(spec)
    intensities = load_intensity_tables(spec, channel="cfse")

    joined = join_ident_with_spots(ident, spots)
    merged = merge_polygons_with_intensity(polygons, intensities)
    arranged, offsets = arrange_rois(merged, max_columns=2, padding=5.0)
    centroids = compute_weighted_centroids(arranged)

    assert joined.height == 4
    assert merged.height == 4
    assert set(offsets) == set(spec.rois)
    assert sorted(centroids["roi"].unique().to_list()) == sorted(spec.rois)
