import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from fishtools.io.workspace import Workspace
from fishtools.preprocess.deconv import normalize


def _write_hist_csv(directory: Path, round_name: str, rows: list[tuple[int, float, float, int]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{round_name}-0001.histogram.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("channel,bin_left,bin_right,count\n")
        for channel, bin_left, bin_right, count in rows:
            handle.write(f"{channel},{bin_left},{bin_right},{count}\n")


def test_precompute_applies_name_overrides_when_tile_lacks_channel_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    round_name = "wga_brdu"
    roi = "roi1"
    override_dir = tmp_path / "analysis" / "deconv32" / f"{round_name}--{roi}"

    # Per-channel histograms for two channels (0 -> WGA, 1 -> BrdU)
    rows = [
        (0, 0.0, 1.0, 10),
        (0, 1.0, 2.0, 5),
        (1, 0.0, 1.0, 8),
        (1, 1.0, 2.0, 4),
    ]
    _write_hist_csv(override_dir, round_name, rows)

    # Float32 tile lacks explicit channel names; axes are reported as ZYX.
    tile_path = override_dir / f"{round_name}-0001.tif"
    data = np.zeros((4, 2, 2), dtype=np.float32)
    metadata = {
        "axes": "ZYX",
        "waveform": json.dumps({"params": {"powers": {"ilm488": 18.0, "ilm750": 8.0}}}),
    }
    tifffile.imwrite(tile_path, data, metadata=metadata)

    # Override mapping uses normalized channel keys (wga -> 0.90, brdu -> 0.85)
    monkeypatch.setattr(
        normalize,
        "P_HIGH_NAME_OVERRIDES_GLOBAL",
        {"wga": 0.90, "brdu": 0.85},
        raising=False,
    )

    # Workspace helper should recover channel names from round/tile tokens
    ws = Workspace(tmp_path)
    assert ws.infer_channel_names(round_name) == ["wga", "brdu"]

    # Scalar default intentionally differs from overrides to expose mismatches
    normalize.precompute(tmp_path, round_name, bins=32, p_low=0.001, p_high=0.5, gamma=1.0)

    meta_path = tmp_path / "analysis" / "deconv32" / "deconv_scaling" / f"{round_name}.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert metadata["p_high_used"] == pytest.approx([0.90, 0.85])
