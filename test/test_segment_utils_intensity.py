import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "fishtools" / "segment" / "utils.py"
_MODULE_SPEC = importlib.util.spec_from_file_location("segment_utils_testmodule", _MODULE_PATH)
if _MODULE_SPEC is None or _MODULE_SPEC.loader is None:  # pragma: no cover - defensive guardrail
    raise RuntimeError(f"Unable to load module spec for {_MODULE_PATH}")
segment_utils = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = segment_utils
_MODULE_SPEC.loader.exec_module(segment_utils)

StitchPaths = segment_utils.StitchPaths
resolve_intensity_store = segment_utils.resolve_intensity_store


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    (ws / "analysis/deconv/stitch--roi+seg").mkdir(parents=True)
    (ws / "workspace.DONE").touch()
    return ws


def test_resolver_accepts_codebook_label(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    target = ws / "analysis/deconv/stitch--roi+edu/fused.zarr"
    target.mkdir(parents=True)
    stitch = StitchPaths.from_workspace(ws, "roi", "seg")

    resolved = resolve_intensity_store(stitch, "edu")

    assert resolved == target


def test_resolver_falls_back_to_sanitized_label(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    target = ws / "analysis/deconv/stitch--roi+edu_test/fused.zarr"
    target.mkdir(parents=True)
    stitch = StitchPaths.from_workspace(ws, "roi", "seg")

    resolved = resolve_intensity_store(stitch, "edu-test")

    assert resolved == target


def test_resolver_errors_for_missing_codebook(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path)
    stitch = StitchPaths.from_workspace(ws, "roi", "seg")

    with pytest.raises(FileNotFoundError):
        resolve_intensity_store(stitch, "unknown")
