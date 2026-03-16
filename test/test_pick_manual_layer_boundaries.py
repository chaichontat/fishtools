from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/mclust/pick_manual_layer_boundaries.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pick_manual_layer_boundaries", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_jobs_from_obs_keeps_rois_separate() -> None:
    module = _load_module()

    obs = pd.DataFrame(
        {
            "dataset": ["ds1", "ds1", "ds1", "ds2"],
            "roi": ["4", "6", "2", "4"],
            "roi_group": ["4", "6", "2", "4"],
            "ccf_adjusted": ["cortex", "cortex", "cortex", "cortex2"],
        }
    )

    jobs = module.build_jobs_from_obs(obs)

    assert [(job.dataset, job.roi_group, job.ccf_adjusted, job.roi_members) for job in jobs] == [
        ("ds1", "2", "cortex", ("2",)),
        ("ds1", "4", "cortex", ("4",)),
        ("ds1", "6", "cortex", ("6",)),
        ("ds2", "4", "cortex2", ("4",)),
    ]


def test_save_and_load_boundary_state_round_trip(tmp_path: Path) -> None:
    module = _load_module()

    path = tmp_path / "job.json"
    job = module.JobSpec(
        dataset="ds1",
        roi_group="4",
        ccf_adjusted="cortex",
        roi_members=("4",),
        n_obs=10,
    )
    boundaries = module.empty_boundaries()
    boundaries["46-2"] = [(1.0, 2.0), (3.0, 4.0)]
    boundaries["2-3"] = [(5.0, 6.0), (7.0, 8.0), (9.0, 10.0)]

    module.save_boundary_state(path=path, job=job, boundaries=boundaries, completed=True)
    loaded = module.load_boundary_state(path)

    assert loaded == boundaries


def test_job_output_stem_keeps_existing_filename_format() -> None:
    module = _load_module()

    job = module.JobSpec(
        dataset="20250929_JaxA3_Coro4",
        roi_group="4",
        ccf_adjusted="cortex",
        roi_members=("4",),
        n_obs=10,
    )

    assert module._job_output_stem(job) == "20250929_JaxA3_Coro4__roi-4__ccf-cortex"


def test_densify_open_curve_keeps_closed_loops_within_anchor_bounds() -> None:
    module = _load_module()

    points = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [3.0, 4.0],
            [1.0, 5.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    dense = module.densify_open_curve(points)

    assert dense[:, 0].min() >= points[:, 0].min()
    assert dense[:, 0].max() <= points[:, 0].max()
    assert dense[:, 1].min() >= points[:, 1].min()
    assert dense[:, 1].max() <= points[:, 1].max()


def test_load_adata_with_labels_replaces_stale_obs_columns(tmp_path: Path) -> None:
    module = _load_module()

    obs = pd.DataFrame(
        {
            "dataset": ["ds1", "ds2"],
            "roi": ["4", "2"],
            "ccf_adjusted": ["cortex", "cortex"],
            "mclust_k7": ["stale", "stale"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        },
        index=pd.Index(["ds1:cell_a", "ds2:cell_a"], name="index"),
    )
    adata = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32), obs=obs)
    in_h5ad = tmp_path / "input.h5ad"
    adata.write_h5ad(in_h5ad)

    labels = pd.DataFrame(
        {
            "mclust_k7": pd.Categorical(["4", "2"]),
            "mclust_k6": pd.Categorical(["4", "2"]),
        },
        index=pd.Index(["ds1:cell_a", "ds2:cell_a"], name="cell_id"),
    )
    labels_parquet = tmp_path / "labels.parquet"
    labels.to_parquet(labels_parquet)

    loaded = module.load_adata_with_labels(in_h5ad=in_h5ad, labels_parquet=labels_parquet)

    assert loaded.obs["mclust_k7"].astype(str).tolist() == ["4", "2"]
    assert str(loaded.obs["mclust_k7"].dtype) == "category"
    assert loaded.obs["obs_ix"].tolist() == [0, 1]
    assert loaded.obs["roi_group"].tolist() == ["4", "2"]


def test_assign_manual_layers_from_boundaries_partitions_ordered_bands() -> None:
    module = _load_module()

    xs = np.repeat(np.arange(1.0, 60.0, 2.0), 15)
    ys = np.tile(np.arange(0.0, 30.0, 2.0), 30)
    xy = np.column_stack([xs, ys]).astype(np.float32)

    expected_layers = np.empty(xy.shape[0], dtype=object)
    expected_layers[xy[:, 0] < 10.0] = "46"
    expected_layers[(xy[:, 0] >= 10.0) & (xy[:, 0] < 20.0)] = "2"
    expected_layers[(xy[:, 0] >= 20.0) & (xy[:, 0] < 30.0)] = "3"
    expected_layers[(xy[:, 0] >= 30.0) & (xy[:, 0] < 40.0)] = "7"
    expected_layers[(xy[:, 0] >= 40.0) & (xy[:, 0] < 50.0)] = "5"
    expected_layers[xy[:, 0] >= 50.0] = "1"

    original_labels = expected_layers.copy()
    original_labels[expected_layers == "46"] = np.where(ys[expected_layers == "46"] % 4 == 0, "4", "6")

    boundaries = {
        "46-2": [(10.0, 4.0), (10.0, 24.0)],
        "2-3": [(20.0, 4.0), (20.0, 24.0)],
        "3-7": [(30.0, 4.0), (30.0, 24.0)],
        "7-5": [(40.0, 4.0), (40.0, 24.0)],
        "5-1": [(50.0, 4.0), (50.0, 24.0)],
    }

    result = module.assign_manual_layers_from_boundaries(
        xy=xy,
        original_labels=original_labels,
        boundaries=boundaries,
        pixel_size=1.0,
        support_radius=1,
        min_component_pixels=10,
    )

    assert result.component_order == tuple(sorted(result.component_order))
    assert result.manual_layers.tolist() == expected_layers.tolist()
    assert sorted(np.unique(result.manual_layers).tolist()) == ["1", "2", "3", "46", "5", "7"]


def test_assign_manual_layers_allows_multiple_components_per_layer() -> None:
    module = _load_module()

    xs_a = np.repeat(np.arange(1.0, 60.0, 2.0), 15)
    ys_a = np.tile(np.arange(1.0, 30.0, 2.0), 30)
    xs_b = np.repeat(np.arange(1.0, 60.0, 2.0), 15)
    ys_b = np.tile(np.arange(101.0, 130.0, 2.0), 30)
    xy = np.column_stack([np.concatenate([xs_a, xs_b]), np.concatenate([ys_a, ys_b])]).astype(np.float32)

    expected_layers = np.empty(xy.shape[0], dtype=object)
    expected_layers[xy[:, 0] < 10.0] = "46"
    expected_layers[(xy[:, 0] >= 10.0) & (xy[:, 0] < 20.0)] = "2"
    expected_layers[(xy[:, 0] >= 20.0) & (xy[:, 0] < 30.0)] = "3"
    expected_layers[(xy[:, 0] >= 30.0) & (xy[:, 0] < 40.0)] = "7"
    expected_layers[(xy[:, 0] >= 40.0) & (xy[:, 0] < 50.0)] = "5"
    expected_layers[xy[:, 0] >= 50.0] = "1"

    original_labels = expected_layers.copy()
    mask_46 = expected_layers == "46"
    original_labels[mask_46] = np.where(np.arange(mask_46.sum()) % 2 == 0, "4", "6")

    boundaries = {
        "46-2": [(10.0, 4.0), (10.0, 24.0)],
        "2-3": [(20.0, 4.0), (20.0, 24.0)],
        "3-7": [(30.0, 4.0), (30.0, 24.0)],
        "7-5": [(40.0, 4.0), (40.0, 24.0)],
        "5-1": [(50.0, 4.0), (50.0, 24.0)],
    }

    result = module.assign_manual_layers_from_boundaries(
        xy=xy,
        original_labels=original_labels,
        boundaries=boundaries,
        pixel_size=1.0,
        support_radius=1,
        min_component_pixels=10,
    )

    assert result.manual_layers.tolist() == expected_layers.tolist()
    assert result.component_order == tuple(range(1, len(module.MANUAL_LAYER_ORDER) + 1))
    assert result.component_layer_map == {i + 1: label for i, label in enumerate(module.MANUAL_LAYER_ORDER)}


def test_assign_manual_layers_supports_closed_boundary_loops() -> None:
    module = _load_module()

    rng_x, rng_y = np.meshgrid(np.arange(2.0, 58.0, 4.0), np.arange(2.0, 58.0, 4.0))
    xy = np.column_stack([rng_x.ravel(), rng_y.ravel()]).astype(np.float32)
    expected_layers = np.full(xy.shape[0], "46", dtype=object)

    loops = {
        "46-2": [(8.0, 8.0), (8.0, 52.0), (52.0, 52.0), (52.0, 8.0), (8.0, 8.0)],
        "2-3": [(16.0, 16.0), (16.0, 44.0), (44.0, 44.0), (44.0, 16.0), (16.0, 16.0)],
        "3-7": [(24.0, 24.0), (24.0, 36.0), (36.0, 36.0), (36.0, 24.0), (24.0, 24.0)],
        "7-5": [(26.0, 26.0), (26.0, 34.0), (34.0, 34.0), (34.0, 26.0), (26.0, 26.0)],
        "5-1": [(28.0, 28.0), (28.0, 32.0), (32.0, 32.0), (32.0, 28.0), (28.0, 28.0)],
    }
    point_rc, occupied, (x0, y0) = module._rasterize_points(xy, pixel_size=1.0)
    support = module._make_support_mask(occupied, radius=1)
    for boundary_name in module.BOUNDARY_ORDER:
        polygon = module._extended_boundary_curve_rc(
            np.asarray(loops[boundary_name], dtype=np.float32),
            shape=support.shape,
            x0=x0,
            y0=y0,
            pixel_size=1.0,
        )
        inside = module._polygon_contains_points(polygon, point_rc.astype(np.float32, copy=False))
        expected_layers[inside] = boundary_name.split("-")[1]

    original_labels = expected_layers.copy()
    original_labels[expected_layers == "46"] = "4"

    result = module.assign_manual_layers_from_boundaries(
        xy=xy,
        original_labels=original_labels,
        boundaries=loops,
        pixel_size=1.0,
        support_radius=1,
        min_component_pixels=4,
    )

    assert result.manual_layers.tolist() == expected_layers.tolist()


def test_save_job_state_persists_json_before_assignment_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    obs = pd.DataFrame(
        {
            "dataset": ["ds1", "ds1"],
            "roi": ["4", "4"],
            "roi_group": ["4", "4"],
            "ccf_adjusted": ["cortex", "cortex"],
            "obs_ix": [0, 1],
            "mclust_k7": pd.Categorical(["4", "2"]),
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        },
        index=pd.Index(["cell_a", "cell_b"], name="cell_id"),
    )
    adata_job = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32), obs=obs)
    adata_job.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    job = module.JobSpec(
        dataset="ds1",
        roi_group="4",
        ccf_adjusted="cortex",
        roi_members=("4",),
        n_obs=2,
    )
    paths = {
        "json": tmp_path / "job.json",
        "assignments": tmp_path / "job.assignments.parquet",
        "qc_png": tmp_path / "job.qc.png",
    }
    boundaries = module.empty_boundaries()
    boundaries["46-2"] = [(0.0, 0.0), (1.0, 0.0)]
    boundaries["2-3"] = [(0.0, 1.0), (1.0, 1.0)]

    monkeypatch.setattr(module, "assign_manual_layers_from_boundaries", lambda **_: (_ for _ in ()).throw(ValueError("boom")))

    ok, message = module.save_job_state_and_assignment(
        adata_job=adata_job,
        job=job,
        paths=paths,
        boundaries=boundaries,
    )

    payload = json.loads(paths["json"].read_text())
    assert not ok
    assert "Saved boundary JSON" in message
    assert payload["completed"] is False
    assert payload["boundaries"]["46-2"] == [[0.0, 0.0], [1.0, 0.0]]
    assert not paths["assignments"].exists()


def test_write_assignment_outputs_includes_obs_index_column(tmp_path: Path) -> None:
    module = _load_module()

    obs = pd.DataFrame(
        {
            "dataset": ["ds1", "ds1"],
            "roi": ["4", "4"],
            "roi_group": ["4", "4"],
            "ccf_adjusted": ["cortex", "cortex"],
            "obs_ix": [0, 1],
            "mclust_k6": pd.Categorical(["46", "2"]),
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        },
        index=pd.Index(["ds1:cell_a", "ds1:cell_b"], name="cell_id"),
    )
    adata_job = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32), obs=obs)
    adata_job.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    job = module.JobSpec(
        dataset="ds1",
        roi_group="4",
        ccf_adjusted="cortex",
        roi_members=("4",),
        n_obs=2,
    )
    paths = {
        "json": tmp_path / "job.json",
        "assignments": tmp_path / "job.assignments.parquet",
        "qc_png": tmp_path / "job.qc.png",
    }
    result = module.AssignmentResult(
        manual_layers=np.array(["46", "2"], dtype=object),
        point_components=np.array([1, 2], dtype=np.int32),
        component_order=(1, 2),
        component_layer_map={1: "46", 2: "2"},
        components_raster=np.array([[1, 2]], dtype=np.int32),
        support_mask=np.array([[True, True]], dtype=bool),
        barrier_mask=np.array([[False, False]], dtype=bool),
        raster_origin_xy=(0.0, 0.0),
        pixel_size=1.0,
    )

    module.write_assignment_outputs(
        adata_job=adata_job,
        job=job,
        result=result,
        paths=paths,
        boundaries=module.empty_boundaries(),
    )

    frame = pd.read_parquet(paths["assignments"])
    assert frame["index"].tolist() == ["ds1:cell_a", "ds1:cell_b"]
    assert frame["obs_ix"].tolist() == [0, 1]


def test_save_job_boundary_state_only_writes_json(tmp_path: Path) -> None:
    module = _load_module()

    job = module.JobSpec(
        dataset="ds1",
        roi_group="4",
        ccf_adjusted="cortex",
        roi_members=("4",),
        n_obs=10,
    )
    paths = {
        "json": tmp_path / "job.json",
        "assignments": tmp_path / "job.assignments.parquet",
        "qc_png": tmp_path / "job.qc.png",
    }
    boundaries = module.empty_boundaries()
    boundaries["46-2"] = [(1.0, 2.0), (3.0, 4.0)]

    message = module.save_job_boundary_state(job=job, paths=paths, boundaries=boundaries)

    payload = json.loads(paths["json"].read_text())
    assert message == "Saved boundary JSON: job.json"
    assert payload["completed"] is False
    assert payload["boundaries"]["46-2"] == [[1.0, 2.0], [3.0, 4.0]]
    assert not paths["assignments"].exists()
    assert not paths["qc_png"].exists()


def test_merge_small_components_absorbs_sliver_into_neighbor() -> None:
    module = _load_module()

    components = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 2, 0],
            [0, 1, 3, 2, 2, 0],
            [0, 1, 1, 2, 2, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    support = components > 0

    merged = module._merge_small_components(components, support=support, min_size=2)

    assert 3 not in np.unique(merged)
