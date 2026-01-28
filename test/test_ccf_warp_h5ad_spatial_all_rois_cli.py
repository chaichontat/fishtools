from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from fishtools.ccf.cli_warp_h5ad_spatial import main


def _make_workspace(tmp_path: Path, *, rois: list[str]) -> Path:
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    (ws_root / "dummy.DONE").write_text("ok", encoding="utf-8")
    for roi in rois:
        (ws_root / f"1--{roi}").mkdir()
    return ws_root


def _write_h5ad_placeholder(ws_root: Path, *, roi: str) -> None:
    h5ads_dir = ws_root / "analysis" / "output" / "h5ads"
    h5ads_dir.mkdir(parents=True, exist_ok=True)
    (h5ads_dir / f"{roi}.h5ad").write_text("placeholder", encoding="utf-8")


def test_no_roi_runs_all_rois_and_skips_missing_h5ads(tmp_path: Path) -> None:
    ws_root = _make_workspace(tmp_path, rois=["cortex", "hippo"])
    runner = CliRunner(mix_stderr=False)

    res = runner.invoke(main, [str(ws_root)], catch_exceptions=False)

    assert res.exit_code == 0
    assert "[cortex] Skipping: Missing export h5ad" in res.output
    assert "[hippo] Skipping: Missing export h5ad" in res.output
    assert "No ROIs processed (all missing input h5ads)." in res.output


def test_roi_still_errors_when_h5ad_missing(tmp_path: Path) -> None:
    ws_root = _make_workspace(tmp_path, rois=["cortex"])
    runner = CliRunner(mix_stderr=False)

    res = runner.invoke(main, [str(ws_root), "cortex"], catch_exceptions=False)

    assert res.exit_code != 0


def test_no_roi_skips_missing_ccf_summary(tmp_path: Path) -> None:
    ws_root = _make_workspace(tmp_path, rois=["cortex"])
    _write_h5ad_placeholder(ws_root, roi="cortex")
    runner = CliRunner(mix_stderr=False)

    res = runner.invoke(main, [str(ws_root)], catch_exceptions=False)

    assert res.exit_code == 0
    assert "[cortex] Skipping: Missing" in res.output


def test_roi_skips_missing_ccf_summary(tmp_path: Path) -> None:
    ws_root = _make_workspace(tmp_path, rois=["cortex"])
    _write_h5ad_placeholder(ws_root, roi="cortex")
    runner = CliRunner(mix_stderr=False)

    res = runner.invoke(main, [str(ws_root), "cortex"], catch_exceptions=False)

    assert res.exit_code == 0
    assert "[cortex] Skipping: Missing" in res.output
