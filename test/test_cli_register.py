from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from fishtools.preprocess.cli_register import (
    DATA,
    Config,
    Fiducial,
    RegisterConfig,
    _copy_codebook_to_workspace,
    _run,
)
from fishtools.preprocess.cli_register import (
    register as register_cli,
)


def _make_codebook(tmp_path: Path) -> Path:
    cb = tmp_path / "cb.json"
    cb.write_text("{}")
    return cb


def _make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "ws"
    deconv = root / "analysis" / "deconv"
    deconv.mkdir(parents=True)
    (root / "workspace.DONE").write_text("")
    return root, deconv


def test_cli_register_run_invokes_internal(tmp_path: Path, monkeypatch: Any) -> None:
    # Arrange: create minimal workspace and codebook
    _root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    called: dict[str, Any] = {}

    log_calls: list[dict[str, Any]] = []

    def fake_setup_workspace_logging(
        workspace_path: Path,
        *,
        component: str,
        file: str,
        idx: int | None = None,
        debug: bool = False,
        extra: dict[str, Any] | None = None,
        **_: Any,
    ) -> Path:
        log_calls.append({
            "workspace": workspace_path,
            "component": component,
            "file": file,
            "idx": idx,
            "debug": debug,
            "extra": extra or {},
        })
        return workspace_path / "analysis" / "logs" / f"{file}.log"

    # Stub heavy internal pipeline to avoid I/O
    def fake__run(
        path: Path,
        roi: str,
        idx: int,
        *,
        codebook: str | Path,
        reference: str,
        config,
        debug: bool,
        overwrite: bool,
        no_priors: bool,
    ) -> None:  # type: ignore[no-untyped-def]
        called.update({
            "path": path,
            "roi": roi,
            "idx": idx,
            "codebook": Path(codebook),
            "reference": reference,
            "config": config,
            "debug": debug,
            "overwrite": overwrite,
            "no_priors": no_priors,
        })

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)
    monkeypatch.setattr(
        "fishtools.preprocess.cli_register.setup_cli_logging",
        fake_setup_workspace_logging,
    )

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(deconv),
            "42",
            "--codebook",
            str(cb),
            "--roi",
            "roiA",
            "--reference",
            "4_12_20",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    # Verify our stub saw the right parameters
    assert called["path"] == deconv
    assert called["roi"] == "roiA"
    assert called["idx"] == 42
    dest_codebook = deconv / "codebooks" / cb.name
    assert called["reference"] == "4_12_20"
    assert called["overwrite"] is True
    # Defaults: fwhm=4.0 (from click default), threshold=5.0
    cfg = called["config"]
    assert pytest.approx(cfg.registration.fiducial.fwhm, rel=0, abs=1e-6) == 4.0
    assert pytest.approx(cfg.registration.fiducial.threshold, rel=0, abs=1e-6) == 5.0
    # Sanity on other core defaults plumbed through
    assert cfg.registration.crop == 40
    assert cfg.registration.downsample == 1
    assert dest_codebook.exists()
    assert dest_codebook.read_text() == cb.read_text()


def test_cli_register_batch_spawns_subprocess(tmp_path: Path, monkeypatch: Any) -> None:
    # Arrange workspace structure expected by batch
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    # create one input tif path that matches the glob, content not used
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")

    cb = _make_codebook(tmp_path)

    # Fake Workspace returned by cli_register.Workspace
    class _WS:
        def __init__(self, path: Path, *_: Any, **__: Any) -> None:
            path = Path(path)
            if path.name == "deconv" and path.parent.name == "analysis":
                self.path = path.parent.parent
                self._deconved = path
            else:
                self.path = path
                self._deconved = self.path / "analysis" / "deconv"
            self.rois = ["roiA"]
            self.rounds = ["2_10_18"]

        @property
        def deconved(self) -> Path:
            return self._deconved

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)

        class _R:  # Minimal CompletedProcess-like shim
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--overwrite",
            "--threads",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    # Expect one submission for idx 1 in roiA
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:3] == ["preprocess", "register", "run"]
    assert str(base) in argv
    copied_codebook = base / "codebooks" / cb.name
    assert f"--codebook={copied_codebook}" in argv
    assert "--reference" in argv  # ref auto-selected
    # Batch should forward default fwhm/threshold into child command
    assert any(a.startswith("--fwhm=") and float(a.split("=", 1)[1]) == 4.0 for a in argv)
    assert any(a.startswith("--threshold=") and float(a.split("=", 1)[1]) == 6.0 for a in argv)


def test_cli_register_run_respects_cli_overrides(tmp_path: Path, monkeypatch: Any) -> None:
    """Ensure CLI flags override the config passed to _run."""
    _root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    seen: dict[str, Any] = {}

    def fake__run(
        path: Path,
        roi: str,
        idx: int,
        *,
        codebook: str | Path,
        reference: str,
        config,
        debug: bool,
        overwrite: bool,
        no_priors: bool,
    ) -> None:  # type: ignore[no-untyped-def]
        seen.update({"config": config, "roi": roi, "idx": idx, "reference": reference})

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(deconv),
            "7",
            "--codebook",
            str(cb),
            "--roi",
            "roiB",
            "--reference",
            "7_15_23",
            "--threshold",
            "7.5",
            "--fwhm",
            "3.5",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = seen["config"]
    assert pytest.approx(cfg.registration.fiducial.threshold, rel=0, abs=1e-6) == 7.5
    assert pytest.approx(cfg.registration.fiducial.fwhm, rel=0, abs=1e-6) == 3.5
    assert seen["roi"] == "roiB"
    assert seen["idx"] == 7
    assert seen["reference"] == "7_15_23"
    copied = deconv / "codebooks" / cb.name
    assert copied.exists()
    assert copied.read_text() == cb.read_text()


def test_cli_register_run_skips_when_shifts_exist(tmp_path: Path, monkeypatch: Any) -> None:
    _root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    shift_dir = deconv / "shifts--roiA+cb"
    shift_dir.mkdir(parents=True)
    (shift_dir / "shifts-0042.json").write_text("{}")

    called = False

    def fake__run(*_: Any, **__: Any) -> None:  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(deconv),
            "42",
            "--codebook",
            str(cb),
            "--roi",
            "roiA",
            "--reference",
            "4_12_20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called is False
    assert (deconv / "codebooks" / cb.name).exists()


def test_run_internal_returns_early_when_shifts_exist(tmp_path: Path) -> None:
    shift_dir = tmp_path / "shifts--roiA+cb"
    shift_dir.mkdir(parents=True)
    (shift_dir / "shifts-0007.json").write_text("{}")

    codebook_path = _make_codebook(tmp_path)
    reg_dir = tmp_path / "registered--roiA+cb"
    reg_dir.mkdir(parents=True)
    (reg_dir / "reg-0007.tif").write_text("")

    cfg = Config(
        dataPath=str(DATA),
        exclude=None,
        registration=RegisterConfig(
            chromatic_shifts={
                "650": str(DATA / "560to650.txt"),
                "750": str(DATA / "560to750.txt"),
            },
            fiducial=Fiducial(
                use_fft=False,
                fwhm=4.0,
                threshold=6.0,
                priors={},
                overrides={},
                n_fids=2,
            ),
            reference="4_12_20",
            downsample=1,
            crop=40,
            slices=slice(None),
            reduce_bit_depth=0,
            discards=None,
        ),
    )

    _run(
        path=tmp_path,
        roi="roiA",
        idx=7,
        codebook=codebook_path,
        reference="4_12_20",
        config=cfg,
        debug=False,
        overwrite=False,
        no_priors=False,
    )

    assert not (tmp_path / "registered--roiA+cb").exists()


def test_cli_register_batch_skips_existing_shifts(tmp_path: Path, monkeypatch: Any) -> None:
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")

    shift_dir = base / "shifts--roiA+cb"
    shift_dir.mkdir(parents=True)
    (shift_dir / "shifts-0001.json").write_text("{}")

    cb = _make_codebook(tmp_path)
    reg_dir = base / "registered--roiA+cb"
    reg_dir.mkdir(parents=True)
    (reg_dir / "reg-0001.tif").write_text("")

    class _WS:
        def __init__(self, path: Path, *_: Any, **__: Any) -> None:
            path = Path(path)
            if path.name == "deconv" and path.parent.name == "analysis":
                self.path = path.parent.parent
                self._deconved = path
            else:
                self.path = path
                self._deconved = self.path / "analysis" / "deconv"
            self.rois = ["roiA"]
            self.rounds = ["2_10_18"]

        @property
        def deconved(self) -> Path:
            return self._deconved

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], check: bool) -> Any:  # type: ignore[no-untyped-def]
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register.subprocess.run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--threads",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    assert (base / "codebooks" / cb.name).exists()


def test_copy_codebook_is_idempotent(tmp_path: Path) -> None:
    _root, deconv = _make_workspace(tmp_path)
    source = _make_codebook(tmp_path)
    dest = deconv / "codebooks" / source.name
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(source.read_text())

    copied = _copy_codebook_to_workspace(deconv, source)

    assert copied == dest
    assert dest.read_text() == source.read_text()


def test_cli_register_batch_verify_reruns_on_read_failure(tmp_path: Path, monkeypatch: Any) -> None:
    # Arrange workspace and two indices
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")
    (base / "2_10_18--roiA" / "2_10_18-0002.tif").write_text("")

    cb = _make_codebook(tmp_path)

    # Fake Workspace
    class _WS:
        def __init__(self, path: Path, *_: Any, **__: Any) -> None:
            path = Path(path)
            if path.name == "deconv" and path.parent.name == "analysis":
                self.path = path.parent.parent
                self._deconved = path
            else:
                self.path = path
                self._deconved = self.path / "analysis" / "deconv"
            self.rois = ["roiA"]
            self.rounds = ["2_10_18"]

        @property
        def deconved(self) -> Path:
            return self._deconved

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    # Child CLI stub creates placeholder output files
    calls: list[list[str]] = []

    def fake_child(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)
        # Create an output file to be 'read' by the TiffFile stub
        assert argv[:3] == ["preprocess", "register", "run"]
        out_dir = Path(argv[3]) / f"registered--roiA+{Path(argv[5].split('=', 1)[1]).stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = int(argv[4])
        (out_dir / f"reg-{idx:04d}.tif").write_text("stub")

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_child)

    # TiffFile stub: baseline (0001) reads OK; 0002 fails once, then succeeds after rerun
    read_attempts: dict[Path, int] = {}
    baseline_shape = (2, 3, 10, 10)

    class _TF:
        def __init__(self, p: Path, *_: Any, **__: Any) -> None:
            self.p = Path(p)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def asarray(self):  # type: ignore[no-untyped-def]
            read_attempts[self.p] = read_attempts.get(self.p, 0) + 1
            name = self.p.name
            if name.endswith("reg-0001.tif"):
                return np.zeros(baseline_shape, dtype=np.uint16)
            if name.endswith("reg-0002.tif") and read_attempts[self.p] == 1:
                raise IndexError("simulated decode error")
            return np.zeros(baseline_shape, dtype=np.uint16)

    monkeypatch.setattr("fishtools.preprocess.cli_register.TiffFile", _TF)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--threads",
            "1",
            "--verify",
        ],
    )

    assert result.exit_code == 0, result.output
    # Initial calls: two indices, plus one rerun for 0002
    # The rerun must include --overwrite
    indices = [int(c[4]) for c in calls if c[:3] == ["preprocess", "register", "run"]]
    assert indices.count(1) == 1
    assert indices.count(2) == 2  # one original + one rerun
    assert any("--overwrite" in c for c in calls if c[4] == "2")

    # Postconditions: TiffFile was attempted twice for reg-0002
    reg_dir = base / f"registered--roiA+{cb.stem}"
    assert (reg_dir / "reg-0001.tif").exists()
    assert (reg_dir / "reg-0002.tif").exists()
    assert read_attempts[reg_dir / "reg-0002.tif"] >= 2


def test_cli_register_batch_verify_checks_existing_outputs_without_overwrite(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Arrange workspace with two indices; both registered files already exist
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")
    (base / "2_10_18--roiA" / "2_10_18-0002.tif").write_text("")

    cb = _make_codebook(tmp_path)
    reg_dir = base / f"registered--roiA+{cb.stem}"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / "reg-0001.tif").write_text("stub")
    (reg_dir / "reg-0002.tif").write_text("stub")

    # Fake Workspace
    class _WS:
        def __init__(self, path: Path, *_: Any, **__: Any) -> None:
            path = Path(path)
            if path.name == "deconv" and path.parent.name == "analysis":
                self.path = path.parent.parent
                self._deconved = path
            else:
                self.path = path
                self._deconved = self.path / "analysis" / "deconv"
            self.rois = ["roiA"]
            self.rounds = ["2_10_18"]

        @property
        def deconved(self) -> Path:
            return self._deconved

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    # Child CLI should only be called for failing index despite overwrite not provided
    calls: list[list[str]] = []

    def fake_child(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_child)

    # TiffFile stub: 0001 fails once (triggers rerun); 0002 always OK
    attempts: dict[str, int] = {"reg-0001.tif": 0}
    baseline_shape = (1, 1, 4, 4)

    class _TF:
        def __init__(self, p: Path, *_: Any, **__: Any) -> None:
            self.p = Path(p)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        def asarray(self):  # type: ignore[no-untyped-def]
            name = self.p.name
            if name == "reg-0001.tif":
                attempts[name] += 1
                if attempts[name] == 1:
                    raise OSError("simulated read error")
                return np.zeros(baseline_shape, dtype=np.uint16)
            return np.zeros(baseline_shape, dtype=np.uint16)

    monkeypatch.setattr("fishtools.preprocess.cli_register.TiffFile", _TF)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--threads",
            "1",
            "--verify",
        ],
    )

    assert result.exit_code == 0, result.output
    # No initial runs should occur (both outputs existed), but one rerun for 0001
    assert len(calls) == 1
    assert calls[0][:3] == ["preprocess", "register", "run"]
    assert calls[0][4] == "1"
    assert "--overwrite" in calls[0]
