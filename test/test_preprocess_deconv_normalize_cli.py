from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from click.testing import CliRunner

from fishtools.preprocess.cli import main as preprocess


def _write_hist_csv(
    dirpath: Path, round_name: str, roi: str, rows: list[tuple[int, float, float, int]]
) -> None:
    target = dirpath / "analysis" / "deconv32" / f"{round_name}--{roi}"
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / f"{round_name}-0001.histogram.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("channel,bin_left,bin_right,count\n")
        for chan, left, right, count in rows:
            fh.write(f"{chan},{left},{right},{count}\n")


def test_deconv_normalize_precompute(tmp_path: Path) -> None:
    round_name = "pi"
    rows = [
        (0, 0.0, 1.0, 10),
        (0, 1.0, 2.0, 20),
        (1, 0.0, 1.0, 5),
        (1, 1.0, 3.0, 15),
    ]
    _write_hist_csv(tmp_path, round_name, "roiA", rows)

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

    scale_dir = tmp_path / "analysis" / "deconv_scaling"
    txt_path = scale_dir / f"{round_name}.txt"
    png_path = scale_dir / f"{round_name}.hist.png"
    assert txt_path.exists()
    assert png_path.exists()
    loaded = np.loadtxt(txt_path)
    assert loaded.shape == (2, 2)


def test_deconv_normalize_quantize(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    round_name = "1_2"
    roi = "cortex"
    n_fids = 2

    raw_dir = workspace / f"{round_name}--{roi}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    deconv32_dir = workspace / "analysis" / "deconv32" / f"{round_name}--{roi}"
    deconv32_dir.mkdir(parents=True, exist_ok=True)

    scaling_dir = workspace / "analysis" / "deconv_scaling"
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
    float32_payload = deconv_stack.reshape(z_slices * channels, height, width)
    float32_path = deconv32_dir / f"{round_name}-0001.tif"
    tifffile.imwrite(float32_path, float32_payload, dtype=np.float32, metadata={"dtype": "float32"})

    fid_planes = np.arange(n_fids * height * width, dtype=np.uint16).reshape(n_fids, height, width)
    raw_payload = np.concatenate([np.zeros_like(float32_payload, dtype=np.uint16), fid_planes], axis=0)
    raw_metadata = {
        "waveform": json.dumps({"params": {"powers": ["bit001", "bit002"]}}),
        "channel_names": ["wga", "edu"],
    }
    raw_path = raw_dir / float32_path.name
    tifffile.imwrite(raw_path, raw_payload, metadata=raw_metadata)

    runner = CliRunner()
    result = runner.invoke(
        preprocess,
        [
            "deconv",
            "quantize",
            str(workspace),
            round_name,
            "--roi",
            roi,
            "--n-fids",
            str(n_fids),
        ],
    )
    assert result.exit_code == 0, result.output

    output_path = workspace / "analysis" / "deconv" / f"{round_name}--{roi}" / float32_path.name
    assert output_path.exists()
