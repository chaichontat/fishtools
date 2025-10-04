from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from click.testing import CliRunner

from fishtools.preprocess.cli import main as preprocess
from fishtools.preprocess.deconv.normalize import I_MAX, quantize_global


def _write_hist_csv(
    dirpath: Path,
    round_name: str,
    roi: str,
    tile_prefix: str,
    rows: list[tuple[int, float, float, int]],
) -> None:
    target = dirpath / "analysis" / "deconv32" / f"{round_name}--{roi}"
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / f"{tile_prefix}{round_name}-0001.histogram.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("channel,bin_left,bin_right,count\n")
        for chan, left, right, count in rows:
            fh.write(f"{chan},{left},{right},{count}\n")


def test_quantize_cli_produces_uint16_deliverable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    round_name = "1_2"
    roi = "cortex"
    n_fids = 2

    raw_dir = workspace / f"{round_name}--{roi}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    deconv32_dir = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}"
    deconv32_dir.mkdir(parents=True, exist_ok=True)

    scaling_dir = workspace / "analysis" / "deconv32" / "deconv_scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)

    m_glob = np.array([10.0, 100.0], dtype=np.float32)
    s_glob = np.array([1000.0, 500.0], dtype=np.float32)
    np.savetxt(scaling_dir / f"{round_name}.txt", np.vstack([m_glob, s_glob]))

    z_slices = 1
    channels = 2
    height = width = 2
    # Shape (Z, C, H, W)
    deconv_stack = np.array(
        [
            [
                [[10.0, 12.0], [11.0, 13.0]],
                [[100.0, 105.0], [110.0, 120.0]],
            ]
        ],
        dtype=np.float32,
    )

    float32_deconv = deconv_stack.reshape(z_slices * channels, height, width)
    fid_planes = np.arange(n_fids * height * width, dtype=np.uint16).reshape(n_fids, height, width)
    raw_metadata = {
        "waveform": json.dumps({"params": {"powers": ["bit001", "bit002"]}}),
        "custom": 42,
    }
    float32_payload = np.concatenate([float32_deconv, fid_planes.astype(np.float32)], axis=0)
    float32_path = deconv32_dir / f"{round_name}-0001.tif"
    tifffile.imwrite(
        float32_path,
        float32_payload,
        dtype=np.float32,
        metadata=raw_metadata | {"dtype": "float32", "axes": "ZYX"},
        photometric="minisblack",
    )
    np.testing.assert_array_equal(tifffile.imread(float32_path)[-n_fids:], fid_planes.astype(np.float32))

    raw_payload = np.concatenate([np.zeros_like(float32_deconv, dtype=np.uint16), fid_planes], axis=0)
    raw_path = raw_dir / float32_path.name
    tifffile.imwrite(
        raw_path,
        raw_payload,
        metadata=raw_metadata | {"axes": "ZYX"},
        photometric="minisblack",
    )

    from fishtools.preprocess.deconv.normalize import quantize as quantize_func

    quantize_func(
        workspace,
        round_name,
        rois=(roi,),
        n_fids=n_fids,
        overwrite=True,
    )

    output_path = workspace / "analysis" / "deconv" / f"{round_name}--{roi}" / float32_path.name
    assert output_path.exists()

    written_stack = tifffile.imread(output_path)
    deconv_from_disk = float32_deconv.reshape(z_slices, channels, height, width)
    expected_quantized = quantize_global(
        deconv_from_disk,
        m_glob,
        s_glob,
        i_max=I_MAX,
        return_stats=False,
        as_numpy=True,
    )
    quantized_written = written_stack[:-n_fids]
    np.testing.assert_allclose(quantized_written, expected_quantized, atol=16)
    np.testing.assert_allclose(written_stack[-n_fids:], fid_planes, atol=3)

    with tifffile.TiffFile(output_path) as tif:
        written_meta = tif.shaped_metadata[0]  # type: ignore[index]

    assert written_meta["prenormalized"] is True
    assert written_meta["deconv_min"] == [float(x) for x in m_glob]
    assert written_meta["deconv_scale"] == [float(x) for x in s_glob]
    assert written_meta["custom"] == raw_metadata["custom"]
    assert written_meta["waveform"] == raw_metadata["waveform"]


def test_precompute_cli_writes_scaling_artifacts(tmp_path: Path) -> None:
    round_name = "pi"
    rows = [
        (0, 0.0, 1.0, 10),
        (0, 1.0, 2.0, 20),
        (1, 0.0, 1.0, 5),
        (1, 1.0, 3.0, 15),
    ]
    _write_hist_csv(tmp_path, round_name, "roiA", "tile-", rows)

    runner = CliRunner()
    result = runner.invoke(
        preprocess,
        [
            "deconv",
            "precompute",
            str(tmp_path),
            round_name,
            "--bins",
            "32",
            "--p-low",
            "0.01",
            "--p-high",
            "0.99",
            "--gamma",
            "1.0",
            "--i-max",
            "65535",
        ],
    )
    assert result.exit_code == 0, result.output

    scale_dir = tmp_path / "analysis" / "deconv32" / "deconv_scaling"
    txt_path = scale_dir / f"{round_name}.txt"
    png_path = scale_dir / f"{round_name}.hist.png"
    assert txt_path.exists()
    assert png_path.exists()
    loaded = np.loadtxt(txt_path)
    assert loaded.shape == (2, 2)


def test_quantize_cli_reads_only_fid_planes(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    round_name = "3_4"
    roi = "ca1"
    n_fids = 3

    raw_dir = workspace / f"{round_name}--{roi}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    deconv32_dir = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}"
    deconv32_dir.mkdir(parents=True, exist_ok=True)

    scaling_dir = workspace / "analysis" / "deconv32" / "deconv_scaling"
    scaling_dir.mkdir(parents=True, exist_ok=True)

    m_glob = np.array([1.0, 2.0], dtype=np.float32)
    s_glob = np.array([10.0, 20.0], dtype=np.float32)
    np.savetxt(scaling_dir / f"{round_name}.txt", np.vstack([m_glob, s_glob]))

    z_slices = 2
    channels = 2
    height = width = 4
    deconv_stack = np.arange(z_slices * channels * height * width, dtype=np.float32).reshape(
        z_slices, channels, height, width
    )
    float32_deconv = deconv_stack.reshape(z_slices * channels, height, width)
    fid_planes = np.full((n_fids, height, width), 42, dtype=np.uint16)
    raw_metadata = {"custom": "meta"}
    float32_payload = np.concatenate([float32_deconv, fid_planes.astype(np.float32)], axis=0)
    float32_path = deconv32_dir / f"{round_name}-0002.tif"
    tifffile.imwrite(
        float32_path,
        float32_payload,
        dtype=np.float32,
        metadata=raw_metadata | {"dtype": "float32", "axes": "ZYX"},
        photometric="minisblack",
    )
    np.testing.assert_array_equal(tifffile.imread(float32_path)[-n_fids:], fid_planes.astype(np.float32))

    raw_payload = np.concatenate([np.zeros_like(float32_deconv, dtype=np.uint16), fid_planes], axis=0)
    raw_path = raw_dir / float32_path.name
    tifffile.imwrite(
        raw_path,
        raw_payload,
        metadata=raw_metadata | {"axes": "ZYX"},
        photometric="minisblack",
    )

    raw_calls = 0
    float32_calls = 0

    original_tiff_asarray = tifffile.TiffFile.asarray

    def _tiff_asarray(self, *args, **kwargs):  # type: ignore[override]
        nonlocal raw_calls, float32_calls
        tif_path = Path(getattr(getattr(self, "filehandle", None), "path", ""))
        if tif_path == raw_path:
            raw_calls += 1
        elif tif_path == float32_path:
            float32_calls += 1
        return original_tiff_asarray(self, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffFile, "asarray", _tiff_asarray)

    from fishtools.preprocess.deconv.normalize import quantize as quantize_func

    quantize_func(
        workspace,
        round_name,
        rois=(roi,),
        n_fids=n_fids,
        overwrite=True,
    )

    assert raw_calls == 0
    assert float32_calls > 0

    quantized_stack = tifffile.imread(
        workspace / "analysis" / "deconv" / f"{round_name}--{roi}" / float32_path.name
    )
    np.testing.assert_allclose(quantized_stack[-n_fids:], fid_planes, atol=3)
