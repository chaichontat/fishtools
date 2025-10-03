import os
from pathlib import Path

import numpy as np

from fishtools.preprocess.deconv.normalize import precompute_global_quantization


def _write_hist_csv(dirpath: Path, round_name: str, roi: str, tile: str, rows: list[tuple[int, float, float, int]]):
    d = dirpath / "analysis" / "deconv32" / f"{round_name}--{roi}"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{tile}{round_name}-0001.histogram.csv"
    with fp.open("w") as f:
        # headerless to test that path too
        for c, bl, br, cnt in rows:
            f.write(f"{c},{bl},{br},{cnt}\n")


def test_precompute_global_quantization_creates_log_histogram(tmp_path: Path):
    # Arrange: two tiny CSVs for two channels (0 and 1)
    round_name = "pi"
    rows1 = [
        (0, 0.0, 1.0, 10),
        (0, 1.0, 2.0, 20),
        (1, 0.0, 1.0, 5),
        (1, 1.0, 3.0, 15),
    ]
    rows2 = [
        (0, 0.0, 1.0, 8),
        (0, 1.0, 2.0, 12),
        (1, 0.0, 1.0, 7),
        (1, 1.0, 3.0, 9),
    ]
    _write_hist_csv(tmp_path, round_name, "roiA", "a-", rows1)
    _write_hist_csv(tmp_path, round_name, "roiB", "b-", rows2)

    # Act: compute global quantization with small bin count for speed
    m_glob, s_glob = precompute_global_quantization(
        tmp_path, round_name, bins=64, p_low=0.01, p_high=0.99, gamma=1.0
    )

    # Assert: histogram PNG exists and is non-empty
    out_png = tmp_path / "analysis" / "deconv32" / "deconv_scaling" / f"{round_name}.hist.png"
    assert out_png.exists() and out_png.stat().st_size > 0

    # Light sanity: returned shapes match channels discovered (2)
    assert m_glob.shape == (2,) and s_glob.shape == (2,)


def test_precompute_global_quantization_multichannel_plot(tmp_path: Path):
    # Arrange: one CSV with 4 channels; include negative bins for clipping path
    round_name = "pi"
    rows = [
        # ch 0 (all positive)
        (0, 0.5, 1.0, 4),
        (0, 1.0, 2.0, 6),
        # ch 1 (mixed)
        (1, -1.0, 0.0, 5),
        (1, 0.0, 2.0, 10),
        # ch 2 (mostly negative; leaves few positive mids)
        (2, -3.0, -1.0, 7),
        (2, -1.0, 0.5, 3),
        # ch 3 (all negative to trigger "no >0 bins")
        (3, -2.0, -1.0, 8),
        (3, -1.0, -0.5, 2),
    ]

    # Write one file containing all channels
    _write_hist_csv(tmp_path, round_name, "roiC", "c-", rows)

    # Act
    m_glob, s_glob = precompute_global_quantization(
        tmp_path, round_name, bins=64, p_low=0.01, p_high=0.99, gamma=1.0
    )

    # Assert
    out_png = tmp_path / "analysis" / "deconv32" / "deconv_scaling" / f"{round_name}.hist.png"
    assert out_png.exists() and out_png.stat().st_size > 0
    assert m_glob.shape == (4,) and s_glob.shape == (4,)
