from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import polars as pl
import pytest

from fishtools.io.workspace import Workspace


def _load_cellpose_stub() -> Callable[[], None]:
    stub_path = Path(__file__).with_name("_cellpose_stub.py")
    spec = importlib.util.spec_from_file_location("_cellpose_stub", stub_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load cellpose stub from {stub_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ensure = getattr(module, "ensure_cellpose_stub", None)
    if ensure is None:
        raise AttributeError("Stub module missing ensure_cellpose_stub")
    return ensure


ensure_cellpose_stub = _load_cellpose_stub()
ensure_cellpose_stub()


def _segment_export_module():  # type: ignore[return-type]
    import fishtools.segment.export as segment_export

    return segment_export


def _export_cmd():
    return _segment_export_module().export_cmd


def _resolve_rois_func():
    return _segment_export_module()._resolve_rois


def _resolve_channels_func():
    return _segment_export_module()._resolve_channels


def _emit_diag_func():
    module = _segment_export_module()
    return module._emit_pairing_diagnostics, module._IntensityKey


def _build_counts_matrix_func():
    return _segment_export_module()._build_counts_matrix


def _annotate_cells_func():
    module = _segment_export_module()
    return module.annotate_cells_with_roi, module.load_roi_polygons


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def _write_segmentation_zarr(path: Path, *, shape: tuple[int, int, int]) -> None:
    import zarr

    path.parent.mkdir(parents=True, exist_ok=True)
    zarr.open_array(path, mode="w", shape=shape, dtype="uint8")


def _write_thumbnail(output_dir: Path, *, size: tuple[int, int]) -> None:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(output_dir / "thumbnail_z000.png")


def _write_roiset_zip(
    output_dir: Path,
    *,
    line: tuple[tuple[float, float], tuple[float, float]],
    polygons: list[tuple[str, list[tuple[float, float]]]],
) -> None:
    from roifile import ImagejRoi, ROI_TYPE, roiwrite

    line_roi = ImagejRoi()
    line_roi.roitype = ROI_TYPE.LINE
    line_roi.name = "line"
    line_roi.x1, line_roi.y1 = line[0]
    line_roi.x2, line_roi.y2 = line[1]
    line_roi.left = int(min(line_roi.x1, line_roi.x2))
    line_roi.right = int(max(line_roi.x1, line_roi.x2))
    line_roi.top = int(min(line_roi.y1, line_roi.y2))
    line_roi.bottom = int(max(line_roi.y1, line_roi.y2))

    roi_entries = [line_roi]
    for name, points in polygons:
        roi_entries.append(ImagejRoi.frompoints(points, name=name))

    roiwrite(output_dir / "RoiSet.zip", roi_entries, mode="w")


def test_segment_export_produces_cells_and_h5ad(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi1"
    seg_codebook = "seg"
    codebook = "gene"
    segmentation_name = "output_segmentation.zarr"
    stitch_root = ws.stitch(roi, seg_codebook)
    seg_root = stitch_root / segmentation_name
    chunks_root = seg_root / f"chunks+{codebook}"

    scale = 8.0
    thumb_size = (10, 8)
    _write_segmentation_zarr(seg_root, shape=(1, int(thumb_size[1] * scale), int(thumb_size[0] * scale)))
    thumb_dir = ws.output / "thumbnails" / f"{roi}+{seg_codebook}"
    _write_thumbnail(thumb_dir, size=thumb_size)

    ident_path = chunks_root / "ident_0000.parquet"
    n_cells = 12
    ident_df = pl.DataFrame({
        "spot_id": list(range(n_cells)),
        "label": list(range(1, n_cells + 1)),
        "target": ["GeneB-1"] * n_cells,
    })
    _write_parquet(ident_path, ident_df)

    polygons_path = chunks_root / "polygons_0000.parquet"
    polygons_df = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "area": [5.0] * n_cells,
        "centroid_x": [10.0 + idx for idx in range(n_cells)],
        "centroid_y": [15.0 + idx for idx in range(n_cells)],
    })
    _write_parquet(polygons_path, polygons_df)

    intensity_root = seg_root / "intensity_marker"
    intensity_path = intensity_root / "intensity-0000.parquet"
    intensity_df = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "mean_intensity": [1.0] * n_cells,
        "median_intensity": [1.0] * n_cells,
        "max_intensity": [1.5] * n_cells,
        "min_intensity": [0.5] * n_cells,
    })
    _write_parquet(intensity_path, intensity_df)

    export_cmd = _export_cmd()
    export_cmd(
        path=ws.deconved,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=(codebook,),
        segmentation_name=segmentation_name,
        channels="marker",
        diag=False,
    )

    cb_token = Workspace.sanitize_codebook_name(codebook)
    seg_stem = Path(segmentation_name).stem
    cells_path = ws.output / f"polygons+{roi}+{cb_token}+{seg_stem}.parquet"
    assert cells_path.exists()
    cells_df = pl.read_parquet(cells_path)
    assert set(["x", "y", "roi", "area", "marker_mean"]).issubset(set(cells_df.columns))
    assert cells_df.schema["area"] == pl.Float32
    assert cells_df.schema["x"] == pl.Float32
    assert cells_df.schema["y"] == pl.Float32
    assert cells_df.schema["marker_mean"] == pl.Float32
    assert cells_df.schema["marker_median"] == pl.Float32
    assert cells_df.schema["marker_max"] == pl.Float32
    assert cells_df.schema["marker_min"] == pl.Float32

    h5ad_path = ws.output / "h5ads" / f"{roi}.h5ad"
    assert h5ad_path.exists()
    adata = ad.read_h5ad(h5ad_path)
    assert adata.n_obs == n_cells
    assert {"marker_mean"}.issubset(set(adata.obs.columns))
    assert {"marker_median"}.issubset(set(adata.obs.columns))
    assert adata.n_vars == 1
    assert {"GeneB"}.issubset(set(adata.var_names))

    assert "fishtools" in adata.uns
    assert "segment_export" in adata.uns["fishtools"]
    assert "qc_filtering" in adata.uns["fishtools"]
    meta = adata.uns["fishtools"]["segment_export"]
    assert meta["workspace_path"] == str(ws.path)
    assert meta["args"]["seg_codebook"] == seg_codebook
    assert meta["resolved"]["rois"] == [roi]
    assert meta["outputs"]["h5ad_path"] == str(h5ad_path)
    assert codebook in set(meta["artifacts_by_roi"][roi]["available_chunks"])
    assert "marker" in set(meta["artifacts_by_roi"][roi]["available_intensity_channels"])
    assert adata.uns["fishtools"]["qc_filtering"]["filter_cells"]["max_counts"] == 1200

    baysor_path = ws.deconved / "baysor" / "spots.csv"
    assert not baysor_path.exists()


def test_segment_export_pools_intensity_std_across_zs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi1"
    seg_codebook = "seg"
    codebook = "gene"
    segmentation_name = "output_segmentation.zarr"
    stitch_root = ws.stitch(roi, seg_codebook)
    seg_root = stitch_root / segmentation_name
    chunks_root = seg_root / f"chunks+{codebook}"

    scale = 8.0
    thumb_size = (10, 8)
    _write_segmentation_zarr(seg_root, shape=(2, int(thumb_size[1] * scale), int(thumb_size[0] * scale)))
    thumb_dir = ws.output / "thumbnails" / f"{roi}+{seg_codebook}"
    _write_thumbnail(thumb_dir, size=thumb_size)

    ident_path = chunks_root / "ident_0000.parquet"
    n_cells = 12
    ident_df = pl.DataFrame({
        "spot_id": list(range(n_cells)),
        "label": list(range(1, n_cells + 1)),
        "target": ["GeneB-1"] * n_cells,
    })
    _write_parquet(ident_path, ident_df)

    polygons_path0 = chunks_root / "polygons_0000.parquet"
    polygons_path1 = chunks_root / "polygons_0001.parquet"
    polygons_df0 = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "area": [1.0] * n_cells,
        "centroid_x": [10.0 + idx for idx in range(n_cells)],
        "centroid_y": [15.0 + idx for idx in range(n_cells)],
    })
    polygons_df1 = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "area": [3.0] * n_cells,
        "centroid_x": [12.0 + idx for idx in range(n_cells)],
        "centroid_y": [18.0 + idx for idx in range(n_cells)],
    })
    _write_parquet(polygons_path0, polygons_df0)
    _write_parquet(polygons_path1, polygons_df1)

    intensity_root = seg_root / "intensity_marker"
    intensity_path0 = intensity_root / "intensity-0000.parquet"
    intensity_path1 = intensity_root / "intensity-0001.parquet"
    intensity_df0 = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "mean_intensity": [0.0] * n_cells,
        "intensity_std": [0.0] * n_cells,
        "max_intensity": [0.0] * n_cells,
        "min_intensity": [0.0] * n_cells,
    })
    intensity_df1 = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "mean_intensity": [2.0] * n_cells,
        "intensity_std": [0.0] * n_cells,
        "max_intensity": [2.0] * n_cells,
        "min_intensity": [2.0] * n_cells,
    })
    _write_parquet(intensity_path0, intensity_df0)
    _write_parquet(intensity_path1, intensity_df1)

    export_cmd = _export_cmd()
    export_cmd(
        path=ws.deconved,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=(codebook,),
        segmentation_name=segmentation_name,
        channels="marker",
        diag=False,
    )

    cb_token = Workspace.sanitize_codebook_name(codebook)
    seg_stem = Path(segmentation_name).stem
    cells_path = ws.output / f"polygons+{roi}+{cb_token}+{seg_stem}.parquet"
    cells_df = pl.read_parquet(cells_path)
    assert "marker_std" in cells_df.columns
    assert cells_df.schema["marker_std"] == pl.Float32
    assert "marker_punctate" in cells_df.columns
    assert cells_df.schema["marker_punctate"] == pl.Float32

    expected = float(np.sqrt(0.75))
    std_value = float(cells_df.filter(pl.col("roilabel") == f"{roi}|1")["marker_std"][0])
    assert std_value == pytest.approx(expected, rel=1e-4, abs=1e-4)

    punct_value = float(cells_df.filter(pl.col("roilabel") == f"{roi}|1")["marker_punctate"][0])
    assert punct_value == pytest.approx(1.0 / 9.0, rel=1e-4, abs=1e-4)

    h5ad_path = ws.output / "h5ads" / f"{roi}.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    assert "marker_std" in adata.obs.columns
    assert "marker_punctate" in adata.obs.columns


def test_segment_export_pools_median_across_zs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi1"
    seg_codebook = "seg"
    codebook = "gene"
    segmentation_name = "output_segmentation.zarr"
    stitch_root = ws.stitch(roi, seg_codebook)
    seg_root = stitch_root / segmentation_name
    chunks_root = seg_root / f"chunks+{codebook}"

    scale = 8.0
    thumb_size = (10, 8)
    _write_segmentation_zarr(seg_root, shape=(2, int(thumb_size[1] * scale), int(thumb_size[0] * scale)))
    thumb_dir = ws.output / "thumbnails" / f"{roi}+{seg_codebook}"
    _write_thumbnail(thumb_dir, size=thumb_size)

    ident_path = chunks_root / "ident_0000.parquet"
    n_cells = 12
    ident_df = pl.DataFrame({
        "spot_id": list(range(n_cells)),
        "label": list(range(1, n_cells + 1)),
        "target": ["GeneB-1"] * n_cells,
    })
    _write_parquet(ident_path, ident_df)

    polygons_path0 = chunks_root / "polygons_0000.parquet"
    polygons_path1 = chunks_root / "polygons_0001.parquet"
    polygons_df0 = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "area": [1.0] * n_cells,
        "centroid_x": [10.0 + idx for idx in range(n_cells)],
        "centroid_y": [15.0 + idx for idx in range(n_cells)],
    })
    polygons_df1 = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "area": [3.0] * n_cells,
        "centroid_x": [12.0 + idx for idx in range(n_cells)],
        "centroid_y": [18.0 + idx for idx in range(n_cells)],
    })
    _write_parquet(polygons_path0, polygons_df0)
    _write_parquet(polygons_path1, polygons_df1)

    intensity_root = seg_root / "intensity_marker"
    intensity_path0 = intensity_root / "intensity-0000.parquet"
    intensity_path1 = intensity_root / "intensity-0001.parquet"
    intensity_df0 = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "mean_intensity": [0.0] * n_cells,
        "median_intensity": [0.0] * n_cells,
        "max_intensity": [0.0] * n_cells,
        "min_intensity": [0.0] * n_cells,
    })
    intensity_df1 = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "mean_intensity": [2.0] * n_cells,
        "median_intensity": [2.0] * n_cells,
        "max_intensity": [2.0] * n_cells,
        "min_intensity": [2.0] * n_cells,
    })
    _write_parquet(intensity_path0, intensity_df0)
    _write_parquet(intensity_path1, intensity_df1)

    export_cmd = _export_cmd()
    export_cmd(
        path=ws.deconved,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=(codebook,),
        segmentation_name=segmentation_name,
        channels="marker",
        diag=False,
    )

    cb_token = Workspace.sanitize_codebook_name(codebook)
    seg_stem = Path(segmentation_name).stem
    cells_path = ws.output / f"polygons+{roi}+{cb_token}+{seg_stem}.parquet"
    cells_df = pl.read_parquet(cells_path)
    assert "marker_median" in cells_df.columns

    median_value = float(cells_df.filter(pl.col("roilabel") == f"{roi}|1")["marker_median"][0])
    assert median_value == pytest.approx(2.0)

    h5ad_path = ws.output / "h5ads" / f"{roi}.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    assert "marker_median" in adata.obs.columns


def test_segment_export_keeps_cells_without_spots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi1"
    seg_codebook = "seg"
    codebook = "gene"
    segmentation_name = "output_segmentation.zarr"
    stitch_root = ws.stitch(roi, seg_codebook)
    seg_root = stitch_root / segmentation_name
    chunks_root = seg_root / f"chunks+{codebook}"

    scale = 8.0
    thumb_size = (10, 8)
    _write_segmentation_zarr(seg_root, shape=(1, int(thumb_size[1] * scale), int(thumb_size[0] * scale)))
    thumb_dir = ws.output / "thumbnails" / f"{roi}+{seg_codebook}"
    _write_thumbnail(thumb_dir, size=thumb_size)

    n_cells = 12
    n_spotted = 10

    ident_path = chunks_root / "ident_0000.parquet"
    ident_df = pl.DataFrame({
        "spot_id": list(range(n_spotted)),
        "label": list(range(1, n_spotted + 1)),
        "target": ["GeneB-1"] * n_spotted,
    })
    _write_parquet(ident_path, ident_df)

    polygons_path = chunks_root / "polygons_0000.parquet"
    polygons_df = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "area": [5.0] * n_cells,
        "centroid_x": [10.0 + idx for idx in range(n_cells)],
        "centroid_y": [15.0 + idx for idx in range(n_cells)],
    })
    _write_parquet(polygons_path, polygons_df)

    export_cmd = _export_cmd()
    export_cmd(
        path=ws.deconved,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=(codebook,),
        segmentation_name=segmentation_name,
        channels="auto",
        diag=False,
    )

    h5ad_path = ws.output / "h5ads" / f"{roi}.h5ad"
    adata = ad.read_h5ad(h5ad_path)
    assert adata.n_obs == n_cells
    assert f"{roi}|{n_cells}" in set(adata.obs_names)
    assert adata[adata.obs_names == f"{roi}|{n_cells}", :].X.sum() == 0


def test_segment_export_roiset_rotation_and_subroi(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi1"
    seg_codebook = "seg"
    codebook = "gene"
    segmentation_name = "output_segmentation.zarr"

    stitch_root = ws.stitch(roi, seg_codebook)
    seg_root = stitch_root / segmentation_name
    chunks_root = seg_root / f"chunks+{codebook}"

    scale = 8.0
    thumb_size = (10, 8)
    _write_segmentation_zarr(seg_root, shape=(1, int(thumb_size[1] * scale), int(thumb_size[0] * scale)))
    thumb_dir = ws.output / "thumbnails" / f"{roi}+{seg_codebook}"
    _write_thumbnail(thumb_dir, size=thumb_size)

    ident_path = chunks_root / "ident_0000.parquet"
    n_cells = 12
    ident_df = pl.DataFrame({
        "spot_id": list(range(n_cells)),
        "label": list(range(1, n_cells + 1)),
        "target": ["GeneB-1"] * n_cells,
    })
    _write_parquet(ident_path, ident_df)

    line_p0 = (2.0, 2.0)
    line_p1 = (8.0, 4.0)

    polygons_path = chunks_root / "polygons_0000.parquet"
    centroid_x = [line_p0[0] * scale, line_p1[0] * scale] + [100.0] * (n_cells - 2)
    centroid_y = [line_p0[1] * scale, line_p1[1] * scale] + [100.0] * (n_cells - 2)
    polygons_df = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "area": [5.0] * n_cells,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
    })
    _write_parquet(polygons_path, polygons_df)

    intensity_root = seg_root / "intensity_marker"
    intensity_path = intensity_root / "intensity-0000.parquet"
    intensity_df = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "mean_intensity": [1.0] * n_cells,
        "max_intensity": [1.5] * n_cells,
        "min_intensity": [0.5] * n_cells,
    })
    _write_parquet(intensity_path, intensity_df)

    _write_roiset_zip(
        thumb_dir,
        line=(line_p0, line_p1),
        polygons=[("subA", [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])],
    )

    export_cmd = _export_cmd()
    export_cmd(
        path=ws.deconved,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=(codebook,),
        segmentation_name=segmentation_name,
        channels="marker",
        diag=False,
    )

    out_h5ad = ws.output / "h5ads" / f"{roi}.h5ad"
    adata = ad.read_h5ad(out_h5ad)

    obs_p0 = adata.obs.loc[f"{roi}|1"]
    obs_p1 = adata.obs.loc[f"{roi}|2"]
    assert obs_p0["subroi"] == "subA"
    assert obs_p1["subroi"] == ""

    # Canonical x/y stay in fused coordinates for downstream indexing.
    assert abs(float(obs_p0["y"]) - float(obs_p1["y"])) > 1e-3

    # RoiSet line rotation is exposed separately for visualization.
    assert "spatial_roiset" in adata.obsm
    assert "x_roiset" in adata.obs.columns
    assert "y_roiset" in adata.obs.columns
    assert abs(float(obs_p0["y_roiset"]) - float(obs_p1["y_roiset"])) < 1e-3


def test_segment_export_roiset_multiple_lines_errors(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi1"
    seg_codebook = "seg"
    codebook = "gene"
    segmentation_name = "output_segmentation.zarr"

    stitch_root = ws.stitch(roi, seg_codebook)
    seg_root = stitch_root / segmentation_name
    chunks_root = seg_root / f"chunks+{codebook}"

    thumb_size = (10, 8)
    scale = 8.0
    _write_segmentation_zarr(seg_root, shape=(1, int(thumb_size[1] * scale), int(thumb_size[0] * scale)))
    thumb_dir = ws.output / "thumbnails" / f"{roi}+{seg_codebook}"
    _write_thumbnail(thumb_dir, size=thumb_size)

    ident_path = chunks_root / "ident_0000.parquet"
    n_cells = 12
    ident_df = pl.DataFrame({
        "spot_id": list(range(n_cells)),
        "label": list(range(1, n_cells + 1)),
        "target": ["GeneB-1"] * n_cells,
    })
    _write_parquet(ident_path, ident_df)

    polygons_path = chunks_root / "polygons_0000.parquet"
    polygons_df = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "area": [5.0] * n_cells,
        "centroid_x": [16.0] * n_cells,
        "centroid_y": [16.0] * n_cells,
    })
    _write_parquet(polygons_path, polygons_df)

    intensity_root = seg_root / "intensity_marker"
    intensity_path = intensity_root / "intensity-0000.parquet"
    intensity_df = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "mean_intensity": [1.0] * n_cells,
        "max_intensity": [1.5] * n_cells,
        "min_intensity": [0.5] * n_cells,
    })
    _write_parquet(intensity_path, intensity_df)

    from roifile import ImagejRoi, ROI_TYPE, roiwrite

    line1 = ImagejRoi()
    line1.roitype = ROI_TYPE.LINE
    line1.name = "line1"
    line1.x1, line1.y1, line1.x2, line1.y2 = (1.0, 1.0, 8.0, 4.0)
    line1.left = 1
    line1.right = 8
    line1.top = 1
    line1.bottom = 4

    line2 = ImagejRoi()
    line2.roitype = ROI_TYPE.LINE
    line2.name = "line2"
    line2.x1, line2.y1, line2.x2, line2.y2 = (2.0, 2.0, 7.0, 5.0)
    line2.left = 2
    line2.right = 7
    line2.top = 2
    line2.bottom = 5

    roiwrite(thumb_dir / "RoiSet.zip", [line1, line2], mode="w")

    export_cmd = _export_cmd()
    with pytest.raises(ValueError, match="Expected exactly one line ROI"):
        export_cmd(
            path=ws.deconved,
            roi=roi,
            seg_codebook=seg_codebook,
            codebooks=(codebook,),
            segmentation_name=segmentation_name,
            channels="marker",
            diag=False,
        )


def test_segment_export_runs_without_intensity_data(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi1"
    seg_codebook = "seg"
    codebook = "gene"
    segmentation_name = "output_segmentation.zarr"

    stitch_root = ws.stitch(roi, seg_codebook)
    seg_root = stitch_root / segmentation_name
    chunks_root = seg_root / f"chunks+{codebook}"

    scale = 8.0
    thumb_size = (10, 8)
    _write_segmentation_zarr(seg_root, shape=(1, int(thumb_size[1] * scale), int(thumb_size[0] * scale)))
    thumb_dir = ws.output / "thumbnails" / f"{roi}+{seg_codebook}"
    _write_thumbnail(thumb_dir, size=thumb_size)

    n_cells = 12
    ident_path = chunks_root / "ident_0000.parquet"
    ident_df = pl.DataFrame({
        "spot_id": list(range(n_cells)),
        "label": list(range(1, n_cells + 1)),
        "target": ["GeneB-1"] * n_cells,
    })
    _write_parquet(ident_path, ident_df)

    polygons_path = chunks_root / "polygons_0000.parquet"
    polygons_df = pl.DataFrame({
        "label": list(range(1, n_cells + 1)),
        "area": [5.0] * n_cells,
        "centroid_x": [10.0 + idx for idx in range(n_cells)],
        "centroid_y": [15.0 + idx for idx in range(n_cells)],
    })
    _write_parquet(polygons_path, polygons_df)

    export_cmd = _export_cmd()
    export_cmd(
        path=ws.deconved,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=(codebook,),
        segmentation_name=segmentation_name,
        channels="auto",
        diag=False,
    )

    cb_token = Workspace.sanitize_codebook_name(codebook)
    seg_stem = Path(segmentation_name).stem
    cells_path = ws.output / f"polygons+{roi}+{cb_token}+{seg_stem}.parquet"
    assert cells_path.exists()
    cells_df = pl.read_parquet(cells_path)
    assert set(["x", "y", "roi", "area"]).issubset(set(cells_df.columns))
    assert "marker_mean" not in cells_df.columns

    h5ad_path = ws.output / "h5ads" / f"{roi}.h5ad"
    assert h5ad_path.exists()
    adata = ad.read_h5ad(h5ad_path)
    assert adata.n_obs == n_cells
    assert "marker_mean" not in adata.obs.columns
    assert adata.n_vars == 1
    assert {"GeneB"}.issubset(set(adata.var_names))


def test_segment_export_missing_ident_error_lists_available_chunks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "1"
    seg_codebook = "pi"
    segmentation_name = "output_segmentation.zarr"

    seg_root = ws.stitch(roi, seg_codebook) / segmentation_name
    seg_root.mkdir(parents=True, exist_ok=True)
    (seg_root / "chunks+cs_base").mkdir(parents=True, exist_ok=True)

    export_cmd = _export_cmd()
    with pytest.raises(ValueError) as excinfo:
        export_cmd(
            path=ws.deconved,
            roi=roi,
            seg_codebook=seg_codebook,
            codebooks=("pi",),
            segmentation_name=segmentation_name,
            channels="auto",
            diag=False,
        )

    message = str(excinfo.value)
    assert "ident_*.parquet" in message
    assert "cs_base" in message
    assert "overlay spots" in message


def test_segment_export_multi_codebook_writes_distinct_h5ad_name(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    ws.deconved.mkdir(parents=True, exist_ok=True)

    roi = "roi1"
    seg_codebook = "seg"
    segmentation_name = "output_segmentation.zarr"
    seg_root = ws.stitch(roi, seg_codebook) / segmentation_name

    primary_codebook = "gene"
    secondary_codebook = "gene2"
    chunks_primary = seg_root / f"chunks+{primary_codebook}"
    chunks_secondary = seg_root / f"chunks+{secondary_codebook}"

    n_cells = 12
    _write_parquet(
        chunks_primary / "ident_0000.parquet",
        pl.DataFrame({
            "spot_id": list(range(n_cells)),
            "label": list(range(1, n_cells + 1)),
            "target": ["GeneB-1"] * n_cells,
        }),
    )
    _write_parquet(
        chunks_secondary / "ident_0000.parquet",
        pl.DataFrame({
            "spot_id": list(range(n_cells, 2 * n_cells)),
            "label": list(range(1, n_cells + 1)),
            "target": ["GeneC-1"] * n_cells,
        }),
    )
    _write_parquet(
        chunks_primary / "polygons_0000.parquet",
        pl.DataFrame({
            "label": list(range(1, n_cells + 1)),
            "area": [5.0] * n_cells,
            "centroid_x": [10.0 + idx for idx in range(n_cells)],
            "centroid_y": [15.0 + idx for idx in range(n_cells)],
        }),
    )

    export_cmd = _export_cmd()
    export_cmd(
        path=ws.deconved,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=(primary_codebook, secondary_codebook),
        segmentation_name=segmentation_name,
        channels="auto",
        diag=False,
    )

    seg_stem = Path(segmentation_name).stem
    cb_token = Workspace.sanitize_codebook_name(primary_codebook)
    cells_path = ws.output / f"polygons+{roi}+{cb_token}+{seg_stem}.parquet"
    assert cells_path.exists()

    out_h5ad = ws.output / "h5ads" / f"{roi}.h5ad"
    assert out_h5ad.exists()
    adata = ad.read_h5ad(out_h5ad)
    assert adata.n_obs == n_cells
    assert {"GeneB", "GeneC"}.issubset(set(adata.var_names))


def test_resolve_rois_defaults_to_workspace_rois(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    for roi in ("roi_a", "roi_b"):
        ws.stitch(roi, "seg").mkdir(parents=True, exist_ok=True)

    resolve_rois = _resolve_rois_func()
    assert resolve_rois(ws, None) == ["roi_a", "roi_b"]
    assert resolve_rois(ws, "roi_a") == ["roi_a"]


def test_resolve_channels_auto_and_manual(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    ws = Workspace(workspace)
    segmentation_name = "output_segmentation.zarr"
    for roi in ("roi_a", "roi_b"):
        base = ws.stitch(roi, "seg") / segmentation_name
        (base / "intensity_marker2").mkdir(parents=True, exist_ok=True)
        (base / "intensity_marker1").mkdir(parents=True, exist_ok=True)

    resolve_channels = _resolve_channels_func()
    auto_channels = resolve_channels(ws, ["roi_a", "roi_b"], "seg", segmentation_name, "auto")
    assert auto_channels == ["marker1", "marker2"]

    manual_channels = resolve_channels(ws, ["roi_a"], "seg", segmentation_name, "marker2 , marker1")
    assert manual_channels == ["marker2", "marker1"]


def test_diag_helper_raises_on_mismatch() -> None:
    emit_diag, intensity_key_cls = _emit_diag_func()
    poly_df = pl.DataFrame({
        "z": [0],
        "label": [1],
        "roi": ["roi1"],
        "area": [1.0],
        "centroid_x": [0.0],
        "centroid_y": [0.0],
        "roilabel": ["roi1|1"],
    })
    polygons = {"roi1": poly_df}
    intensities = {
        intensity_key_cls(roi="roi1", channel="brdu"): pl.DataFrame({
            "z": [0],
            "roi": ["roi1"],
            "label": [2],
            "brdu_mean": [1.0],
            "brdu_max": [1.0],
            "brdu_min": [1.0],
        })
    }
    with pytest.raises(ValueError, match="Segmentation masks disagree"):
        emit_diag(polygons, intensities, ["brdu"])


def test_counts_matrix_matches_transcript_gene_conversion() -> None:
    build_counts = _build_counts_matrix_func()
    targets = ["GeneA-202", "GeneA-201", "GeneX-201", "GeneY-2-204"]
    df = pl.DataFrame({
        "roilabel": ["roi1|1", "roi1|1", "roi1|2", "roi1|3"],
        "target": targets,
    })
    counts = build_counts({("roi1", "cb"): df})

    expected_genes = {"GeneA-202", "GeneA-201", "GeneX", "GeneY-2"}
    result_genes = set(counts.columns) - {"roilabel"}
    assert result_genes == expected_genes


def test_annotate_cells_with_roi_marks_obs_column(tmp_path: Path) -> None:
    annotate_cells, _ = _annotate_cells_func()

    roi_path = tmp_path / "roi.zip"
    _write_roi_zip(roi_path)

    import anndata as ad
    import numpy as np

    adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
    adata.obs["x"] = [40.0, 400.0, -10.0]
    adata.obs["y"] = [40.0, 400.0, -10.0]
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()

    annotate_cells(adata, roi_path, scale=8.0, column_name="roi_mask")

    assert adata.obs["roi_mask"].tolist() == ["1", "", ""]


def _write_roi_zip(path: Path) -> None:
    import zipfile

    import roifile

    ImagejRoi = roifile.ImagejRoi
    roi = ImagejRoi.frompoints([(0, 0), (10, 0), (10, 10), (0, 10)])
    roi.name = "1"
    roi_file = path.with_suffix(".roi")
    roi.tofile(roi_file)
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(roi_file, arcname="1.roi")
    roi_file.unlink()


def _convert_transcript_to_gene(ts: str, non_unique_targets: list[str] | None = None) -> str:
    if non_unique_targets and any(ts.startswith(nt) for nt in non_unique_targets):
        return ts
    gene, _ = ts.rsplit("-", 1)
    return gene
