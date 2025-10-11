from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from fishtools.preprocess.spots.align_prod import find_threshold as find_threshold_cmd


def _write_txt_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr)


def test_find_threshold_writes_percentiles_cpu(tmp_path: Path) -> None:
    """find-threshold should produce percentiles.json from highpass inputs (CPU path)."""

    roi = "roi"
    cb_name = "cb"

    # Layout: registered--{roi}+{cb}/_highpassed/reg-0001_{cb}.hp.tif
    reg_dir = tmp_path / f"registered--{roi}+{cb_name}" / "_highpassed"
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Create a small 4D highpass image (Z, C, Y, X) with constant values
    z, c, y, x = (1, 2, 4, 4)
    value = 5.0
    hp = (np.ones((z, c, y, x), dtype=np.float32) * value)
    hp_path = reg_dir / f"reg-0001_{cb_name}.hp.tif"
    tifffile.imwrite(hp_path, hp)

    # Optimization outputs needed by find_threshold
    opt_dir = tmp_path / f"opt_{cb_name}+{roi}"
    opt_dir.mkdir(parents=True, exist_ok=True)

    # mins and global_scale (two channels)
    _write_txt_array(opt_dir / "global_min.txt", np.array([0.0, 0.0], dtype=np.float32))
    _write_txt_array(opt_dir / "global_scale.txt", np.array([1.0, 1.0], dtype=np.float32))

    # Codebook file path (content unused by find_threshold)
    cb_path = tmp_path / f"{cb_name}.json"
    cb_path.write_text("{}")

    # Run CLI handler directly
    find_threshold_cmd.callback(
        path=tmp_path,
        roi=roi,
        codebook=cb_path,
        overwrite=False,
        round_num=0,
        max_proj=0,
        blank=None,
        json_config=None,
    )

    # Validate output JSON exists and contains our entry
    out_json = opt_dir / "percentiles.json"
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    assert isinstance(data, list) and data, "percentiles.json should be a non-empty list"

    # Expected percentile is sqrt(sum(ch^2)) with channel value = 5.0 after scaling
    expected = float(np.sqrt(2.0) * value)

    flat_map = {k: v for d in data for k, v in d.items()}
    # Key includes parent folder name and filename
    expected_key_suffix = f"registered--{roi}+{cb_name}-{hp_path.name}"
    match = [v for k, v in flat_map.items() if k.endswith(expected_key_suffix)]
    assert match, f"Missing key ending with {expected_key_suffix} in percentiles.json"
    assert abs(match[0] - expected) < 1e-5
