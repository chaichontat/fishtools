from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from click.testing import CliRunner

from fishtools.segment import app as segment_app
from fishtools.io.workspace import Workspace


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "workspace.DONE").touch()
    return ws


def _write_fused(ws: Path, *, roi: str, codebook: str, name: str, shape: tuple[int, int, int, int]) -> Path:
    workspace = Workspace(ws)
    out_dir = workspace.stitch(roi, codebook)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / name
    arr = zarr.open_array(store_path, mode="w", shape=shape, chunks=shape, dtype=np.uint16)
    arr[...] = np.arange(arr.size, dtype=np.uint16).reshape(shape)
    return store_path


def test_segment_thumbnail_respects_z_range_and_overwrite(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_fused(ws, roi="roi", codebook="cb1", name="fused.zarr", shape=(5, 4, 4, 2))

    workspace = Workspace(ws)
    thumb_dir = (workspace.output / "thumbnails") / "roi+cb1"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = thumb_dir / "thumbnail_z001.png"
    sentinel_path.write_bytes(b"sentinel")

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            "roi",
            "--codebook",
            "cb1",
            "--z-stride",
            "1",
            "--z-range",
            "1:3",
            "--downsample",
            "1",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output
    assert sentinel_path.read_bytes() != b"sentinel"
    assert not (thumb_dir / "thumbnail_z000.png").exists()
    assert (thumb_dir / "thumbnail_z002.png").exists()


def test_segment_thumbnail_include_n4(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_fused(ws, roi="roi", codebook="cb1", name="fused.zarr", shape=(2, 4, 4, 1))
    _write_fused(ws, roi="roi", codebook="cb1", name="fused_n4.zarr", shape=(2, 4, 4, 1))

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            "roi",
            "--codebook",
            "cb1",
            "--z-range",
            "0:1",
            "--downsample",
            "1",
            "--include-n4",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output

    workspace = Workspace(ws)
    thumb_dir = (workspace.output / "thumbnails") / "roi+cb1"
    assert (thumb_dir / "thumbnail_z000.png").exists()
    assert (thumb_dir / "thumbnail_n4_z000.png").exists()


def test_segment_thumbnail_includes_highpass_when_present(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_fused(ws, roi="roi", codebook="cb1", name="fused.zarr", shape=(2, 4, 4, 1))
    _write_fused(ws, roi="roi", codebook="cb1", name="fused_highpassed.zarr", shape=(2, 4, 4, 1))

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            "roi",
            "--codebook",
            "cb1",
            "--z-range",
            "0:1",
            "--downsample",
            "1",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output

    workspace = Workspace(ws)
    thumb_dir = (workspace.output / "thumbnails") / "roi+cb1"
    assert (thumb_dir / "thumbnail_z000.png").exists()
    assert (thumb_dir / "thumbnail_highpass_z000.png").exists()


def test_segment_thumbnail_zs_selects_explicit_planes(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_fused(ws, roi="roi", codebook="cb1", name="fused.zarr", shape=(5, 4, 4, 1))

    runner = CliRunner()
    out_dir = tmp_path / "thumb_zs"
    res = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            "roi",
            "--codebook",
            "cb1",
            "--z-stride",
            "1",
            "--z-range",
            "0:5",
            "--zs",
            "1, 4",
            "--downsample",
            "1",
            "--output-dir",
            str(out_dir),
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output

    thumb_dir = out_dir / "roi+cb1"
    assert not (thumb_dir / "thumbnail_z000.png").exists()
    assert (thumb_dir / "thumbnail_z001.png").exists()
    assert not (thumb_dir / "thumbnail_z002.png").exists()
    assert not (thumb_dir / "thumbnail_z003.png").exists()
    assert (thumb_dir / "thumbnail_z004.png").exists()


def test_segment_thumbnail_zs_rejects_out_of_range(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    _write_fused(ws, roi="roi", codebook="cb1", name="fused.zarr", shape=(2, 4, 4, 1))

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            "roi",
            "--codebook",
            "cb1",
            "--zs",
            "2",
            "--downsample",
            "1",
            "--output-dir",
            str(tmp_path / "thumb_zs_oob"),
        ],
        prog_name="segment",
    )
    assert res.exit_code != 0
    msg = res.output
    if res.exception is not None:
        msg += str(res.exception)
    assert "Invalid --zs" in msg


def test_segment_thumbnail_random_percentile_normalization(tmp_path: Path) -> None:
    from PIL import Image

    ws_path = _make_workspace(tmp_path)
    roi = "roi"
    codebook = "cb1"

    workspace = Workspace(ws_path)
    out_dir = workspace.stitch(roi, codebook)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "fused.zarr"

    rng = np.random.default_rng(0)
    arr = zarr.open_array(store_path, mode="w", shape=(1, 64, 64, 1), chunks=(1, 64, 64, 1), dtype=np.uint16)
    arr[...] = rng.integers(10, 110, size=arr.shape, dtype=np.uint16)

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws_path),
            roi,
            "--codebook",
            codebook,
            "--z-range",
            "0:1",
            "--downsample",
            "1",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output

    thumb_dir = (workspace.output / "thumbnails") / f"{roi}+{codebook}"
    png_path = thumb_dir / "thumbnail_z000.png"
    assert png_path.exists()
    img = np.asarray(Image.open(png_path))
    assert int(img.max()) > 0


def test_segment_thumbnail_outlines_from_seg_codebook_default_downsample(tmp_path: Path) -> None:
    from PIL import Image
    from skimage.segmentation import find_boundaries

    ws_path = _make_workspace(tmp_path)
    roi = "roi"
    codebook = "img_cb"
    seg_codebook = "seg_cb"

    workspace = Workspace(ws_path)

    # Image zarr under image codebook
    img_dir = workspace.stitch(roi, codebook)
    img_dir.mkdir(parents=True, exist_ok=True)
    img_store = img_dir / "fused.zarr"
    img = zarr.open_array(img_store, mode="w", shape=(1, 16, 16, 1), chunks=(1, 16, 16, 1), dtype=np.uint16)
    img[...] = np.arange(img.size, dtype=np.uint16).reshape(img.shape) + 10

    # Segmentation mask under seg codebook (different directory)
    seg_dir = workspace.stitch(roi, seg_codebook)
    seg_dir.mkdir(parents=True, exist_ok=True)
    mask_store = seg_dir / "masks.zarr"
    mask = zarr.open_array(mask_store, mode="w", shape=(1, 16, 16), chunks=(1, 16, 16), dtype=np.uint16)
    mask[...] = 0
    mask[0, 4:12, 4:12] = 1

    runner = CliRunner()
    res = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws_path),
            roi,
            "--codebook",
            codebook,
            "--seg-codebook",
            seg_codebook,
            "--segmentation-name",
            "masks.zarr",
            "--z-range",
            "0:1",
        ],
        prog_name="segment",
    )
    assert res.exit_code == 0, res.output

    thumb_dir = (workspace.output / "thumbnails") / f"{roi}+{codebook}"
    raw_path = thumb_dir / "thumbnail_z000.png"
    overlay_path = thumb_dir / "thumbnail_mask_z000.png"
    assert raw_path.exists()
    assert overlay_path.exists()

    raw = np.asarray(Image.open(raw_path))
    overlay = np.asarray(Image.open(overlay_path))
    assert raw.shape[:2] == (8, 8)
    assert overlay.shape[:2] == (8, 8)

    mask_slice = np.asarray(mask[0, ::2, ::2])
    boundaries = find_boundaries(mask_slice, mode="outer")
    assert bool(boundaries.any())
    assert bool((overlay[boundaries] == np.array([128, 128, 128], dtype=np.uint8)).all())
    assert not np.array_equal(raw, overlay)


def test_segment_thumbnail_channels_option_selects_channels(tmp_path: Path) -> None:
    from PIL import Image

    ws = _make_workspace(tmp_path)
    roi = "roi"
    codebook = "cb1"

    workspace = Workspace(ws)
    out_dir = workspace.stitch(roi, codebook)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "fused.zarr"

    yv = np.broadcast_to(np.arange(16, dtype=np.uint16)[:, None], (16, 16))
    xv = np.broadcast_to(np.arange(16, dtype=np.uint16)[None, :], (16, 16))
    ch0 = 10 + xv  # varies with X
    ch1 = 10 + yv  # varies with Y
    ch2 = 10 + xv + yv
    ch3 = 50 + xv + yv
    data = np.stack([ch0, ch1, ch2, ch3], axis=2)[None, ...]  # (1, 16, 16, 4)

    arr = zarr.open_array(store_path, mode="w", shape=data.shape, chunks=data.shape, dtype=np.uint16)
    arr[...] = data
    arr.attrs["key"] = ["ch0", "ch1", "ch2", "ch3"]

    runner = CliRunner()

    out0 = tmp_path / "thumb_ch0"
    res0 = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            roi,
            "--codebook",
            codebook,
            "--z-range",
            "0:1",
            "--downsample",
            "1",
            "--channels",
            "ch0",
            "--output-dir",
            str(out0),
        ],
        prog_name="segment",
    )
    assert res0.exit_code == 0, res0.output

    img0 = np.asarray(Image.open(out0 / f"{roi}+{codebook}" / "thumbnail_channels-ch0_z000.png"))
    assert np.array_equal(img0[:, :, 0], img0[:, :, 1])
    assert np.array_equal(img0[:, :, 0], img0[:, :, 2])
    assert int(img0[0, 1, 0]) > int(img0[1, 0, 0])

    out1 = tmp_path / "thumb_ch1"
    res1 = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            roi,
            "--codebook",
            codebook,
            "--z-range",
            "0:1",
            "--downsample",
            "1",
            "--channels",
            "ch1",
            "--output-dir",
            str(out1),
        ],
        prog_name="segment",
    )
    assert res1.exit_code == 0, res1.output

    img1 = np.asarray(Image.open(out1 / f"{roi}+{codebook}" / "thumbnail_channels-ch1_z000.png"))
    assert np.array_equal(img1[:, :, 0], img1[:, :, 1])
    assert np.array_equal(img1[:, :, 0], img1[:, :, 2])
    assert int(img1[0, 1, 0]) < int(img1[1, 0, 0])

    out01 = tmp_path / "thumb_ch0_ch1"
    res01 = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws),
            roi,
            "--codebook",
            codebook,
            "--z-range",
            "0:1",
            "--downsample",
            "1",
            "--channels",
            "ch0,ch1",
            "--output-dir",
            str(out01),
        ],
        prog_name="segment",
    )
    assert res01.exit_code == 0, res01.output

    img01 = np.asarray(Image.open(out01 / f"{roi}+{codebook}" / "thumbnail_channels-ch0-ch1_z000.png"))
    assert int(img01[:, :, 2].max()) == 0


def test_segment_thumbnail_applies_ccf_pose_after_downsample(tmp_path: Path) -> None:
    import json

    from PIL import Image
    from scipy.ndimage import rotate as ndimage_rotate

    ws_path = _make_workspace(tmp_path)
    roi = "roi"
    codebook = "img_cb"
    seg_codebook = "seg_cb"

    workspace = Workspace(ws_path)

    # Image zarr under image codebook
    img_dir = workspace.stitch(roi, codebook)
    img_dir.mkdir(parents=True, exist_ok=True)
    img_store = img_dir / "fused.zarr"
    img = zarr.open_array(img_store, mode="w", shape=(1, 32, 32, 1), chunks=(1, 32, 32, 1), dtype=np.uint16)
    yv = np.broadcast_to(np.arange(32, dtype=np.uint16)[:, None], (32, 32))
    xv = np.broadcast_to(np.arange(32, dtype=np.uint16)[None, :], (32, 32))
    img[0, :, :, 0] = 10 + xv + (2 * yv)

    # Segmentation mask under seg codebook (different directory)
    seg_dir = workspace.stitch(roi, seg_codebook)
    seg_dir.mkdir(parents=True, exist_ok=True)
    mask_store = seg_dir / "masks.zarr"
    mask = zarr.open_array(mask_store, mode="w", shape=(1, 32, 32), chunks=(1, 32, 32), dtype=np.uint16)
    mask[...] = 0
    mask[0, 10:22, 12:20] = 1

    runner = CliRunner()
    out_no_pose = tmp_path / "thumb_no_pose"
    res_no_pose = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws_path),
            roi,
            "--codebook",
            codebook,
            "--seg-codebook",
            seg_codebook,
            "--segmentation-name",
            "masks.zarr",
            "--z-range",
            "0:1",
            "--downsample",
            "2",
            "--output-dir",
            str(out_no_pose),
        ],
        prog_name="segment",
    )
    assert res_no_pose.exit_code == 0, res_no_pose.output

    base_no_pose = np.asarray(Image.open(out_no_pose / f"{roi}+{codebook}" / "thumbnail_z000.png"))
    overlay_no_pose = np.asarray(Image.open(out_no_pose / f"{roi}+{codebook}" / "thumbnail_mask_z000.png"))

    # Add a CCF pose (rotation + flipX) and rerun to a new output directory.
    ccf_roi_dir = workspace.ccf_transforms(roi)
    ccf_roi_dir.mkdir(parents=True, exist_ok=True)
    (ccf_roi_dir / "p1_similarity.tfm").write_text("Parameters: 1 0 0 0\n", encoding="utf-8")
    (ccf_roi_dir / "p1_landmarks.json").write_text(
        json.dumps(
            {
                "prior_rotation_deg": 30,
                "prior_flip_x": True,
                "fixed_points_cropped_xy": [[0, 0], [1, 1]],
                "moving_points_fullres_xy_in_rotated_crop": [[0, 0], [1, 1]],
                "atlas_crop_bbox": [0, 1, 0, 1],
                "sample_rotated_crop_bbox": [0, 1, 0, 1],
            }
        ),
        encoding="utf-8",
    )

    out_pose = tmp_path / "thumb_pose"
    res_pose = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws_path),
            roi,
            "--codebook",
            codebook,
            "--seg-codebook",
            seg_codebook,
            "--segmentation-name",
            "masks.zarr",
            "--z-range",
            "0:1",
            "--downsample",
            "2",
            "--output-dir",
            str(out_pose),
        ],
        prog_name="segment",
    )
    assert res_pose.exit_code == 0, res_pose.output

    base_pose = np.asarray(Image.open(out_pose / f"{roi}+{codebook}" / "thumbnail_z000.png"))
    overlay_pose = np.asarray(Image.open(out_pose / f"{roi}+{codebook}" / "thumbnail_mask_z000.png"))

    expected_base = base_no_pose[:, ::-1, :]
    expected_base = ndimage_rotate(expected_base, angle=-30.0, reshape=True, order=1, mode="nearest")
    expected_base = np.clip(expected_base, 0, 255).astype(np.uint8)

    expected_overlay = overlay_no_pose[:, ::-1, :]
    expected_overlay = ndimage_rotate(expected_overlay, angle=-30.0, reshape=True, order=1, mode="nearest")
    expected_overlay = np.clip(expected_overlay, 0, 255).astype(np.uint8)

    assert np.array_equal(base_pose, expected_base)
    assert np.array_equal(overlay_pose, expected_overlay)

    out_pose_disabled = tmp_path / "thumb_pose_disabled"
    res_pose_disabled = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws_path),
            roi,
            "--codebook",
            codebook,
            "--seg-codebook",
            seg_codebook,
            "--segmentation-name",
            "masks.zarr",
            "--z-range",
            "0:1",
            "--downsample",
            "2",
            "--no-ccf-rotate",
            "--output-dir",
            str(out_pose_disabled),
        ],
        prog_name="segment",
    )
    assert res_pose_disabled.exit_code == 0, res_pose_disabled.output

    base_pose_disabled = np.asarray(Image.open(out_pose_disabled / f"{roi}+{codebook}" / "thumbnail_z000.png"))
    overlay_pose_disabled = np.asarray(Image.open(out_pose_disabled / f"{roi}+{codebook}" / "thumbnail_mask_z000.png"))
    assert np.array_equal(base_pose_disabled, base_no_pose)
    assert np.array_equal(overlay_pose_disabled, overlay_no_pose)


def test_segment_thumbnail_percentiles_cache_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws_path = _make_workspace(tmp_path)
    roi = "roi"
    codebook = "cb1"

    workspace = Workspace(ws_path)
    out_dir = workspace.stitch(roi, codebook)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "fused.zarr"

    arr = zarr.open_array(store_path, mode="w", shape=(2, 32, 32, 1), chunks=(1, 32, 32, 1), dtype=np.uint16)
    yv = np.broadcast_to(np.arange(32, dtype=np.uint16)[:, None], (32, 32))
    xv = np.broadcast_to(np.arange(32, dtype=np.uint16)[None, :], (32, 32))
    arr[0, :, :, 0] = 100 + xv + yv
    arr[1, :, :, 0] = 200 + xv + yv
    arr.attrs["key"] = ["ch0"]

    runner = CliRunner()
    out1 = tmp_path / "thumb_first"
    res1 = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws_path),
            roi,
            "--codebook",
            codebook,
            "--z-range",
            "0:1",
            "--downsample",
            "1",
            "--output-dir",
            str(out1),
        ],
        prog_name="segment",
    )
    assert res1.exit_code == 0, res1.output

    cache_dir = workspace.output / "thumbnail_percentiles" / f"{roi}+{codebook}"
    cache_files = list(cache_dir.glob("*.npy"))
    assert len(cache_files) == 1

    import fishtools.segment.normalize as normalize

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("sample_percentile should not be called when cache exists")

    monkeypatch.setattr(normalize, "sample_percentile", _boom)

    out2 = tmp_path / "thumb_second"
    res2 = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws_path),
            roi,
            "--codebook",
            codebook,
            "--z-range",
            "0:1",
            "--downsample",
            "1",
            "--output-dir",
            str(out2),
        ],
        prog_name="segment",
    )
    assert res2.exit_code == 0, res2.output


def test_segment_thumbnail_percentiles_cache_is_not_per_channel_combo(tmp_path: Path) -> None:
    ws_path = _make_workspace(tmp_path)
    roi = "roi"
    codebook = "cb1"

    workspace = Workspace(ws_path)
    out_dir = workspace.stitch(roi, codebook)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "fused.zarr"

    arr = zarr.open_array(store_path, mode="w", shape=(1, 32, 32, 2), chunks=(1, 32, 32, 2), dtype=np.uint16)
    yv = np.broadcast_to(np.arange(32, dtype=np.uint16)[:, None], (32, 32))
    xv = np.broadcast_to(np.arange(32, dtype=np.uint16)[None, :], (32, 32))
    arr[0, :, :, 0] = 100 + xv + yv
    arr[0, :, :, 1] = 200 + (2 * xv) + yv
    arr.attrs["key"] = ["ch0", "ch1"]

    runner = CliRunner()
    out1 = tmp_path / "thumb_first"
    res1 = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws_path),
            roi,
            "--codebook",
            codebook,
            "--z-range",
            "0:1",
            "--downsample",
            "1",
            "--channels",
            "ch0",
            "--output-dir",
            str(out1),
        ],
        prog_name="segment",
    )
    assert res1.exit_code == 0, res1.output

    cache_dir = workspace.output / "thumbnail_percentiles" / f"{roi}+{codebook}"
    cache_files = list(cache_dir.glob("*.npy"))
    assert len(cache_files) == 1

    out2 = tmp_path / "thumb_second"
    res2 = runner.invoke(
        segment_app,
        [
            "thumbnail",
            str(ws_path),
            roi,
            "--codebook",
            codebook,
            "--z-range",
            "0:1",
            "--downsample",
            "1",
            "--channels",
            "ch0,ch1",
            "--output-dir",
            str(out2),
        ],
        prog_name="segment",
    )
    assert res2.exit_code == 0, res2.output

    cache_files2 = list(cache_dir.glob("*.npy"))
    assert len(cache_files2) == 2
