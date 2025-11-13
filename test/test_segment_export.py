from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import anndata as ad
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


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_segment_export_produces_cells_and_h5ad(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    analysis_deconv = workspace / "analysis/deconv"
    analysis_deconv.mkdir(parents=True)
    (workspace / "workspace.DONE").touch()

    roi = "roi1"
    seg_codebook = "seg"
    codebook = "gene"
    stitch_root = analysis_deconv / f"stitch--{roi}+{seg_codebook}"
    chunks_root = stitch_root / f"chunks+{codebook}"

    ident_path = chunks_root / "ident_0000.parquet"
    ident_df = pl.DataFrame({
        "spot_id": [0, 1, 2, 3],
        "label": [1, 1, 2, 2],
        "target": ["GeneA-1", "GeneA-2", "GeneB-1", "GeneC-1"],
    })
    _write_parquet(ident_path, ident_df)

    polygons_path = chunks_root / "polygons_0000.parquet"
    polygons_df = pl.DataFrame({
        "label": [1, 2],
        "area": [5.0, 7.0],
        "centroid_x": [10.0, 20.0],
        "centroid_y": [15.0, 25.0],
    })
    _write_parquet(polygons_path, polygons_df)

    intensity_root = stitch_root / "intensity_marker"
    intensity_path = intensity_root / "intensity-0000.parquet"
    intensity_df = pl.DataFrame({
        "label": [1, 2],
        "mean_intensity": [1.0, 2.0],
        "max_intensity": [1.5, 2.5],
        "min_intensity": [0.5, 1.0],
    })
    _write_parquet(intensity_path, intensity_df)

    export_cmd = _export_cmd()
    export_cmd(
        path=analysis_deconv,
        roi=roi,
        seg_codebook=seg_codebook,
        codebooks=(codebook,),
        channels="marker",
        out_dir=None,
        diag=False,
    )

    cells_path = analysis_deconv / "segment_export" / f"cells--{seg_codebook}+{codebook}.parquet"
    assert cells_path.exists()
    cells_df = pl.read_parquet(cells_path)
    assert set(["x", "y", "roi", "area", "marker_mean"]).issubset(set(cells_df.columns))

    h5ad_path = workspace / "analysis/output" / f"{roi}+{codebook}.h5ad"
    assert h5ad_path.exists()
    adata = ad.read_h5ad(h5ad_path)
    assert adata.n_obs == 2
    assert {"marker_mean"}.issubset(set(adata.obs.columns))
    assert {"GeneA-1", "GeneA-2", "GeneB", "GeneC"}.issubset(set(adata.var_names))

    baysor_path = analysis_deconv / "baysor" / "spots.csv"
    assert not baysor_path.exists()


def test_resolve_rois_defaults_to_workspace_rois(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    analysis_deconv = workspace / "analysis/deconv"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    for roi in ("roi_a", "roi_b"):
        (analysis_deconv / f"stitch--{roi}+seg").mkdir(parents=True, exist_ok=True)

    ws = Workspace(analysis_deconv)
    resolve_rois = _resolve_rois_func()
    assert resolve_rois(ws, None) == ["roi_a", "roi_b"]
    assert resolve_rois(ws, "roi_a") == ["roi_a"]


def test_resolve_channels_auto_and_manual(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    analysis_deconv = workspace / "analysis/deconv"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "workspace.DONE").touch()
    for roi in ("roi_a", "roi_b"):
        base = analysis_deconv / f"stitch--{roi}+seg"
        (base / "intensity_marker2").mkdir(parents=True, exist_ok=True)
        (base / "intensity_marker1").mkdir(parents=True, exist_ok=True)

    resolve_channels = _resolve_channels_func()
    auto_channels = resolve_channels(analysis_deconv, ["roi_a", "roi_b"], "seg", "auto")
    assert auto_channels == ["marker1", "marker2"]

    manual_channels = resolve_channels(analysis_deconv, ["roi_a"], "seg", "marker2 , marker1")
    assert manual_channels == ["marker2", "marker1"]


def test_diag_helper_raises_on_mismatch() -> None:
    emit_diag, intensity_key_cls = _emit_diag_func()
    poly_df = pl.DataFrame(
        {
            "z": [0],
            "label": [1],
            "roi": ["roi1"],
            "area": [1.0],
            "centroid_x": [0.0],
            "centroid_y": [0.0],
            "roilabel": ["roi1|1"],
        }
    )
    polygons = {"roi1": poly_df}
    intensities = {
        intensity_key_cls(roi="roi1", channel="brdu"): pl.DataFrame(
            {
                "z": [0],
                "roi": ["roi1"],
                "label": [2],
                "brdu_mean": [1.0],
                "brdu_max": [1.0],
                "brdu_min": [1.0],
            }
        )
    }
    with pytest.raises(ValueError, match="Segmentation masks disagree"):
        emit_diag(polygons, intensities, ["brdu"])
