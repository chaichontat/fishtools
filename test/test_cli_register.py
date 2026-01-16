from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from tifffile import imwrite

import fishtools.preprocess.cli_register as cli_register_module
from fishtools.io.workspace import Workspace
from fishtools.preprocess.cli_register import (
    DATA,
    Config,
    Fiducial,
    RegisterConfig,
    _copy_codebook_to_workspace,
    _debug_fid_paths,
    _load_shifts_from_codebook,
    _run,
    _write_shifts_json,
    _save_debug_overlay,
)
from fishtools.preprocess.cli_register import (
    register as register_cli,
)
from fishtools.preprocess.fiducial import Shifts


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


def _write_config_file(
    tmp_path: Path,
    *,
    chromatic_path: Path | None = None,
    registration: dict[str, Any] | None = None,
    filename: str = "config.json",
) -> Path:
    reg: dict[str, Any] = dict(registration or {})
    if chromatic_path is not None:
        reg["chromatic_path"] = str(chromatic_path)
    path = tmp_path / filename
    path.write_text(json.dumps({"registration": reg}), encoding="utf-8")
    return path


def test_debug_fid_paths_include_roi(tmp_path: Path) -> None:
    base = tmp_path / "ws" / "analysis" / "deconv"
    debug_dir, raw, shifted = _debug_fid_paths(base, "roiA", 7, "cb")
    assert debug_dir == base.parent / "output" / "fids_debug" / "roiA"
    assert raw == "roiA+cb-0007.tif"
    assert shifted == "roiA+cb-shifted-0007.tif"


def test_debug_fid_paths_with_relative_path(tmp_path: Path, monkeypatch: Any) -> None:
    """Ensure debug_dir resolves correctly even when given a relative path like '.'"""
    base = tmp_path / "ws" / "analysis" / "deconv"
    base.mkdir(parents=True)
    monkeypatch.chdir(base)

    # Use "." as the path (simulating running from deconv directory)
    debug_dir, raw, shifted = _debug_fid_paths(Path("."), "roiA", 7, "cb")

    # Should resolve to absolute path under output/fids_debug
    expected = base.parent / "output" / "fids_debug" / "roiA"
    assert debug_dir == expected
    assert "output" in debug_dir.parts
    assert "fids_debug" in debug_dir.parts


def test_save_debug_overlay_prefixes_roi(tmp_path: Path) -> None:
    shifted = {
        "reference": np.ones((6, 6), dtype=np.float32),
        "round2": np.ones((6, 6), dtype=np.float32) * 5,
    }
    _save_debug_overlay(tmp_path, "roiZ", 12, "reference", shifted, codebook_name="cb")
    assert (tmp_path / "roiZ+cb-0012-overlay.png").exists()


def test_workspace_transformations_paths_use_roi_folder(tmp_path: Path) -> None:
    root, _deconv = _make_workspace(tmp_path)
    ws = Workspace(root)
    with pytest.raises(AttributeError):
        _ = ws.register_transformations_shift_json("roiA", "cb-1", 7)


def test_write_shifts_json_writes_only_to_deconv_shifts(tmp_path: Path) -> None:
    root, _deconv = _make_workspace(tmp_path)
    ws = Workspace(root)

    payload = b"{\"ok\": true}\n"
    _write_shifts_json(ws, roi="roiA", codebook="cb", idx=1, payload=payload)

    deconv_path = ws.shifts("roiA", "cb") / "shifts-0001.json"

    assert deconv_path.read_bytes() == payload
    assert not (root / "analysis" / "output" / "transformations").exists()


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
        repaired_rounds: set[str] | None = None,
        max_iters: int = 5,
        use_shifts_from: str | None = None,
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
            "repaired_rounds": repaired_rounds,
            "max_iters": max_iters,
            "use_shifts_from": use_shifts_from,
        })

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)
    monkeypatch.setattr(
        "fishtools.preprocess.cli_register.setup_cli_logging",
        fake_setup_workspace_logging,
    )

    config_path = _write_config_file(
        tmp_path,
        chromatic_path=DATA,
        registration={
            "reference": "4_12_20",
            "fiducial": {"fwhm": 4.0, "threshold": 5.0},
        },
        filename="register_config.json",
    )

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(deconv),
            "42",
            "--config",
            str(config_path),
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
    assert called["reference"] == "4_12_20"
    assert called["overwrite"] is True
    assert called["codebook"].parent == deconv / "codebooks"
    assert called["codebook"].read_bytes() == cb.read_bytes()
    # Defaults: fwhm=4.0 (from click default), threshold=5.0
    cfg = called["config"]
    assert pytest.approx(cfg.registration.fiducial.fwhm, rel=0, abs=1e-6) == 4.0
    assert pytest.approx(cfg.registration.fiducial.threshold, rel=0, abs=1e-6) == 5.0
    # Sanity on other core defaults plumbed through
    assert cfg.registration.crop == 40
    assert cfg.registration.downsample == 1
    assert cfg.registration.fiducial.detailed.use_brightest == 20
    assert cfg.registration.fiducial.detailed.offset_brightest == 0
    assert cfg.registration.fiducial.detailed.allow_large_shifts is False


def test_cli_register_run_accepts_config_file(tmp_path: Path, monkeypatch: Any) -> None:
    _root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    register_config = RegisterConfig(
        fiducial=Fiducial(
            threshold=9.5,
            fwhm=7.25,
            use_fft=True,
            use_itk=True,
        ),
        chromatic_path=DATA,
        reference="2_10_18",
        crop=12,
        downsample=3,
        reduce_bit_depth=1,
    )
    config_path = tmp_path / "register_config.json"
    config_path.write_text(register_config.model_dump_json(), encoding="utf-8")

    called: dict[str, Any] = {}

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
        repaired_rounds: set[str] | None = None,
        max_iters: int = 5,
        use_shifts_from: str | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        called.update({
            "path": path,
            "roi": roi,
            "idx": idx,
            "reference": reference,
            "config": config,
        })

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)
    monkeypatch.setattr("fishtools.preprocess.cli_register.setup_cli_logging", lambda *_a, **_k: deconv)

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
            "--config",
            str(config_path),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["reference"] == "2_10_18"
    cfg = called["config"]
    assert cfg.registration.crop == 12
    assert cfg.registration.downsample == 3
    assert cfg.registration.reduce_bit_depth == 1
    assert pytest.approx(cfg.registration.fiducial.threshold, rel=0, abs=1e-6) == 9.5
    assert pytest.approx(cfg.registration.fiducial.fwhm, rel=0, abs=1e-6) == 7.25
    assert cfg.registration.fiducial.use_fft is True
    assert cfg.registration.fiducial.use_itk is True


def test_cli_register_run_accepts_partial_config_file(tmp_path: Path, monkeypatch: Any) -> None:
    _root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"registration": {"crop": 120}}), encoding="utf-8")

    called: dict[str, Any] = {}

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
        repaired_rounds: set[str] | None = None,
        max_iters: int = 5,
        use_shifts_from: str | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        called.update({"reference": reference, "config": config})

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)
    monkeypatch.setattr("fishtools.preprocess.cli_register.setup_cli_logging", lambda *_a, **_k: deconv)

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
            "--config",
            str(config_path),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = called["config"]
    assert cfg.registration.crop == 120
    assert cfg.registration.fiducial is not None
    assert cfg.registration.chromatic_path is not None


def test_cli_register_run_config_overridden_by_cli(tmp_path: Path, monkeypatch: Any) -> None:
    _root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    register_config = RegisterConfig(
        fiducial=Fiducial(
            threshold=9.5,
            fwhm=7.25,
        ),
        chromatic_path=DATA,
        reference="2_10_18",
    )
    config_path = tmp_path / "register_config.json"
    config_path.write_text(register_config.model_dump_json(), encoding="utf-8")

    called: dict[str, Any] = {}

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
        repaired_rounds: set[str] | None = None,
        max_iters: int = 5,
        use_shifts_from: str | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        called.update({"reference": reference, "config": config})

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)
    monkeypatch.setattr("fishtools.preprocess.cli_register.setup_cli_logging", lambda *_a, **_k: deconv)

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
            "--config",
            str(config_path),
            "--threshold",
            "1.25",
            "--reference",
            "4_12_20",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["reference"] == "4_12_20"
    cfg = called["config"]
    assert pytest.approx(cfg.registration.fiducial.threshold, rel=0, abs=1e-6) == 1.25


def test_cli_register_run_requires_config(tmp_path: Path) -> None:
    _root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(deconv),
            "1",
            "--codebook",
            str(cb),
            "--roi",
            "roiA",
            "--overwrite",
        ],
    )

    assert result.exit_code != 0
    assert "Missing option '--config'" in result.output


def test_cli_register_copies_used_chromatic_corrections_to_output(tmp_path: Path, monkeypatch: Any) -> None:
    root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)
    config_path = _write_config_file(tmp_path, chromatic_path=DATA, filename="register_config.json")

    def fake_setup_workspace_logging(*_: Any, **__: Any) -> Path:
        return deconv / "analysis" / "logs" / "noop.log"

    def fake__run(*_: Any, **__: Any) -> None:
        return None

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
            "--config",
            str(config_path),
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

    ws = Workspace(root)
    for filename in ("560to650.txt", "560to750.txt"):
        copied = ws.output.chromatic / filename
        assert copied.exists()
        assert copied.read_bytes() == (DATA / filename).read_bytes()

    assert (ws.output.chromatic / "chromatic_provenance.json").exists()


def test_cli_register_does_not_overwrite_existing_output_chromatic_files(tmp_path: Path, monkeypatch: Any) -> None:
    root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    ws = Workspace(root)
    chromatic_dir = ws.output.chromatic
    chromatic_dir.mkdir(parents=True, exist_ok=True)
    (chromatic_dir / "560to650.txt").write_bytes(b"old\n")
    (chromatic_dir / "560to750.txt").write_bytes(b"old\n")

    chromatic_source = tmp_path / "chromatic"
    chromatic_source.mkdir(parents=True, exist_ok=True)
    sentinel = b"custom-chromatic-contents\n"
    (chromatic_source / "560to650.txt").write_bytes(sentinel)
    (chromatic_source / "560to750.txt").write_bytes(sentinel)
    config_path = _write_config_file(
        tmp_path,
        chromatic_path=chromatic_source,
        filename="register_config.json",
    )

    def fake_setup_workspace_logging(*_: Any, **__: Any) -> Path:
        return deconv / "analysis" / "logs" / "noop.log"

    def fake__run(*_: Any, **__: Any) -> None:
        return None

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
            "--config",
            str(config_path),
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
    assert (ws.output.chromatic / "560to650.txt").read_bytes() == sentinel
    assert (ws.output.chromatic / "560to750.txt").read_bytes() == sentinel


def test_cli_register_overwrites_mismatched_output_chromatic_files(tmp_path: Path, monkeypatch: Any) -> None:
    root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)
    config_path = _write_config_file(tmp_path, chromatic_path=DATA, filename="register_config.json")

    ws = Workspace(root)
    chromatic_dir = ws.output.chromatic
    chromatic_dir.mkdir(parents=True, exist_ok=True)
    (chromatic_dir / "560to650.txt").write_bytes(b"old\n")
    (chromatic_dir / "560to750.txt").write_bytes(b"old\n")

    def fake_setup_workspace_logging(*_: Any, **__: Any) -> Path:
        return deconv / "analysis" / "logs" / "noop.log"

    def fake__run(*_: Any, **__: Any) -> None:
        return None

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
            "--config",
            str(config_path),
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
    assert (ws.output.chromatic / "560to650.txt").read_bytes() == (DATA / "560to650.txt").read_bytes()
    assert (ws.output.chromatic / "560to750.txt").read_bytes() == (DATA / "560to750.txt").read_bytes()
    assert (ws.output.chromatic / "chromatic_provenance.json").exists()


def test_load_chromatic_affines_prefers_output_chromatic_dir(tmp_path: Path) -> None:
    root, _deconv = _make_workspace(tmp_path)
    ws = Workspace(root)

    out_dir = ws.output.chromatic
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "560to650.txt").write_text("1 2 3 4 5 6\n", encoding="utf-8")
    (out_dir / "560to750.txt").write_text("7 8 9 10 11 12\n", encoding="utf-8")

    As, ats, meta = cli_register_module._load_chromatic_affines(ws)
    assert meta["650"]["source"] == str(out_dir / "560to650.txt")
    assert meta["750"]["source"] == str(out_dir / "560to750.txt")
    assert meta["650"]["A"] == [[1.0, 2.0], [3.0, 4.0]]
    assert meta["650"]["t"] == [5.0, 6.0]
    assert meta["750"]["A"] == [[7.0, 8.0], [9.0, 10.0]]
    assert meta["750"]["t"] == [11.0, 12.0]
    assert set(As.keys()) == {"650", "750"}
    assert set(ats.keys()) == {"650", "750"}


def test_register_writes_chromatic_values_to_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    _root, deconv = _make_workspace(tmp_path)
    roi = "roiA"
    idx = 0
    round_name = "1_2"

    round_dir = deconv / f"{round_name}--{roi}"
    round_dir.mkdir(parents=True, exist_ok=True)

    h = w = 32
    payload = np.zeros((2, h, w), dtype=np.uint16)
    fid0 = np.random.default_rng(0).integers(0, 1000, size=(h, w), dtype=np.uint16)
    fid1 = np.random.default_rng(1).integers(0, 1000, size=(h, w), dtype=np.uint16)
    stack = np.concatenate([payload, fid0[None, ...], fid1[None, ...]], axis=0)

    waveform = {
        "ilm405": {"sequence": [0], "power": 0.0},
        "ilm488": {"sequence": [0], "power": 0.0},
        "ilm560": {"sequence": [1], "power": 0.0},
        "ilm650": {"sequence": [1], "power": 0.0},
        "ilm750": {"sequence": [0], "power": 0.0},
        "params": {"powers": ["ilm560", "ilm650"]},
    }
    imwrite(
        round_dir / f"{round_name}-{idx:04d}.tif",
        stack,
        metadata={"waveform": json.dumps(waveform), "prenormalized": True},
    )

    cb = tmp_path / "cb.json"
    cb.write_text(json.dumps({"geneX": ["2"]}))

    ws = Workspace(deconv)
    shifts_path = ws.shift_json(roi, "src", idx)
    shifts_path.parent.mkdir(parents=True, exist_ok=True)
    shifts_path.write_text(
        json.dumps({round_name: {"shifts": [0.0, 0.0], "corr": 1.0, "residual": 0.0}})
    )

    chromatic_meta = {
        "650": {"source": "dummy-650", "A": [[1.0, 0.0], [0.0, 1.0]], "t": [2.0, 3.0]},
        "750": {"source": "dummy-750", "A": [[1.0, 0.0], [0.0, 1.0]], "t": [4.0, 5.0]},
        "ref": {"channel": "560"},
    }

    A = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)

    monkeypatch.setattr(
        cli_register_module,
        "_load_chromatic_affines",
        lambda _ws=None: ({"650": A, "750": A}, {"650": t, "750": t}, chromatic_meta),
    )

    class _DummyAffine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.ref_image = None

        def __call__(
            self,
            img: np.ndarray,
            *,
            channel: str,
            shiftpx: np.ndarray,
            debug: bool = False,
        ) -> np.ndarray:
            return img

    monkeypatch.setattr(cli_register_module, "Affine", _DummyAffine)

    captured: dict[str, Any] = {}

    def fake_safe_imwrite(*args: Any, **kwargs: Any) -> None:
        captured["metadata"] = kwargs.get("metadata")

    monkeypatch.setattr(cli_register_module, "safe_imwrite", fake_safe_imwrite)

    cfg = Config()
    cfg = cfg.model_copy(update={"registration": cfg.registration.model_copy(update={"crop": 0})})

    _run(
        deconv,
        roi,
        idx,
        codebook=cb,
        reference=round_name,
        config=cfg,
        overwrite=True,
        debug=False,
        use_shifts_from="src",
    )

    metadata = captured["metadata"]
    assert metadata is not None
    assert json.loads(metadata["chromatic"]) == chromatic_meta



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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

        def registered(self, roi: str, codebook: str) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}"

        def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"shifts--{roi}+{codebook}" / f"shifts-{idx:04d}.json"

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)

        class _R:  # Minimal CompletedProcess-like shim
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    config_path = _write_config_file(
        tmp_path,
        chromatic_path=DATA,
        registration={"reference": "2_10_18"},
        filename="register_config.json",
    )

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--config",
            str(config_path),
            "--codebook",
            str(cb),
            "--overwrite",
            "--threads",
            "1",
            "--allow-large-shifts",
            "--offset-brightest",
            "10",
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
    assert f"--config={config_path}" in argv
    assert "--reference" not in argv
    assert not any(a.startswith("--fwhm=") for a in argv)
    assert not any(a.startswith("--threshold=") for a in argv)
    assert "--allow-large-shifts" in argv
    assert "--offset-brightest=10" in argv


def test_cli_register_batch_forwards_config_file_without_overrides(tmp_path: Path, monkeypatch: Any) -> None:
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")

    cb = _make_codebook(tmp_path)
    register_config = RegisterConfig(
        fiducial=Fiducial(threshold=10.0, fwhm=9.0),
        chromatic_path=DATA,
        reference="2_10_18",
    )
    config_path = tmp_path / "register_config.json"
    config_path.write_text(register_config.model_dump_json(), encoding="utf-8")

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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

        def registered(self, roi: str, codebook: str) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}"

        def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"shifts--{roi}+{codebook}" / f"shifts-{idx:04d}.json"

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--config",
            str(config_path),
            "--codebook",
            str(cb),
            "--overwrite",
            "--threads",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    argv = calls[0]
    assert f"--config={config_path}" in argv
    assert "--reference" not in argv
    assert not any(a.startswith("--fwhm=") for a in argv)
    assert not any(a.startswith("--threshold=") for a in argv)
    assert not any(a.startswith("--use-brightest=") for a in argv)


def test_cli_register_batch_forwards_debug(tmp_path: Path, monkeypatch: Any) -> None:
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")

    cb = _make_codebook(tmp_path)
    config_path = _write_config_file(
        tmp_path,
        chromatic_path=DATA,
        registration={"reference": "2_10_18"},
        filename="register_config.json",
    )

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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--config",
            str(config_path),
            "--codebook",
            str(cb),
            "--overwrite",
            "--threads",
            "1",
            "--debug",
        ],
    )

    assert result.exit_code == 0, result.output
    register_calls = [call for call in calls if call[:3] == ["preprocess", "register", "run"]]
    assert len(register_calls) == 1
    assert "--debug" in register_calls[0]


def test_cli_register_batch_only_median_gt_requires_overwrite(tmp_path: Path, monkeypatch: Any) -> None:
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")

    cb = _make_codebook(tmp_path)
    config_path = _write_config_file(
        tmp_path,
        chromatic_path=DATA,
        registration={"reference": "2_10_18"},
        filename="register_config.json",
    )

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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--config",
            str(config_path),
            "--codebook",
            str(cb),
            "--threads",
            "1",
            "--only-median-gt",
            "1.0",
        ],
    )

    assert result.exit_code != 0
    assert "--only-median-gt requires --overwrite" in result.output


def test_cli_register_batch_only_median_gt_filters_tiles(tmp_path: Path, monkeypatch: Any) -> None:
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    for idx in (1, 2, 3):
        (base / "2_10_18--roiA" / f"2_10_18-{idx:04d}.tif").write_text("")

    cb = _make_codebook(tmp_path)

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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

        def registered(self, roi: str, codebook: str) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}"

        def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"shifts--{roi}+{codebook}" / f"shifts-{idx:04d}.json"

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)

        if argv[:2] == ["preprocess", "check-shifts"]:
            out_dir = Path(argv[argv.index("--output") + 1])
            roi_value = argv[3]
            codebook_path = Path(argv[argv.index("--codebook") + 1])
            csv_path = out_dir / "shifts_metrics" / f"shifts_metrics--{roi_value}+{codebook_path.stem}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("tile,L2\n1,0.1\n2,2.1\n3,5.0\n")

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
            "--overwrite",
            "--threads",
            "1",
            "--only-median-gt",
            "2.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert sum(1 for call in calls if call[:2] == ["preprocess", "check-shifts"]) == 2

    register_calls = [call for call in calls if call[:3] == ["preprocess", "register", "run"]]
    assert len(register_calls) == 2
    seen_idxs = sorted(int(call[4]) for call in register_calls)
    assert seen_idxs == [2, 3]


def test_cli_register_batch_only_corr_lt_requires_overwrite(tmp_path: Path, monkeypatch: Any) -> None:
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")

    cb = _make_codebook(tmp_path)

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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
            "--threads",
            "1",
            "--only-corr-lt",
            "0.5",
        ],
    )

    assert result.exit_code != 0
    assert "--only-corr-lt requires --overwrite" in result.output


def test_cli_register_batch_only_corr_lt_filters_tiles(tmp_path: Path, monkeypatch: Any) -> None:
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    for idx in (1, 2, 3):
        (base / "2_10_18--roiA" / f"2_10_18-{idx:04d}.tif").write_text("")

    cb = _make_codebook(tmp_path)

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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

        def registered(self, roi: str, codebook: str) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}"

        def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"shifts--{roi}+{codebook}" / f"shifts-{idx:04d}.json"

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)

        if argv[:2] == ["preprocess", "check-shifts"]:
            out_dir = Path(argv[argv.index("--output") + 1])
            roi_value = argv[3]
            codebook_path = Path(argv[argv.index("--codebook") + 1])
            csv_path = out_dir / "shifts_metrics" / f"shifts_metrics--{roi_value}+{codebook_path.stem}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("tile,correlation\n1,0.95\n2,0.05\n3,0.20\n")

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
            "--overwrite",
            "--threads",
            "1",
            "--only-corr-lt",
            "0.3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert sum(1 for call in calls if call[:2] == ["preprocess", "check-shifts"]) == 2

    register_calls = [call for call in calls if call[:3] == ["preprocess", "register", "run"]]
    assert len(register_calls) == 2
    seen_idxs = sorted(int(call[4]) for call in register_calls)
    assert seen_idxs == [2, 3]


def test_cli_register_batch_falls_back_to_fids_for_idx_discovery(tmp_path: Path, monkeypatch: Any) -> None:
    _root, base = _make_workspace(tmp_path)
    # No ref round directory created: fallback must use fids--roiA to determine indices.
    fid_dir = base / "fids--roiA"
    fid_dir.mkdir(parents=True, exist_ok=True)
    imwrite(fid_dir / "fids-0002.tif", np.zeros((4, 4), dtype=np.uint16))
    imwrite(fid_dir / "fids-0007.tif", np.zeros((4, 4), dtype=np.uint16))

    cb = _make_codebook(tmp_path)

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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

        def registered(self, roi: str, codebook: str) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}"

        def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"shifts--{roi}+{codebook}" / f"shifts-{idx:04d}.json"

        def fids(self, roi: str) -> Path:
            return self._deconved / f"fids--{roi}"

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
            "--overwrite",
            "--threads",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    register_calls = [call for call in calls if call[:3] == ["preprocess", "register", "run"]]
    seen_idxs = sorted(int(call[4]) for call in register_calls)
    assert seen_idxs == [2, 7]


def test_load_reference_fid_falls_back_to_fids_dir(tmp_path: Path) -> None:
    root, deconv = _make_workspace(tmp_path)
    ws = cli_register_module.Workspace(root)

    roi = "roiA"
    idx = 7
    reference = "2_10_18"

    fid_dir = deconv / f"fids--{roi}"
    fid_dir.mkdir(parents=True, exist_ok=True)
    expected = np.arange(16, dtype=np.float32).reshape(4, 4)
    imwrite(fid_dir / f"fids-{idx:04d}.tif", expected)

    loaded = cli_register_module._load_reference_fid_from_previous_run(
        ws,
        roi=roi,
        reference=reference,
        idx=idx,
        prefer_codebook=None,
    )

    assert np.allclose(loaded, expected)


def test_run_fiducial_debug_prints_priors(tmp_path: Path, monkeypatch: Any) -> None:
    _root, deconv = _make_workspace(tmp_path)

    cfg = Config(
        dataPath=str(DATA),
        exclude=None,
        registration=RegisterConfig(
            chromatic_path=DATA,
            fiducial=Fiducial(
                use_fft=False,
                fwhm=4.0,
                threshold=6.0,
                priors={"round_b": (1.0, 2.0)},
                overrides={},
                n_fids=2,
            ),
            reference="round_a",
            downsample=1,
            crop=0,
            slices=slice(None),
            reduce_bit_depth=0,
            discards=None,
        ),
    )

    def fake_align_with_stats(
        fids: dict[str, np.ndarray], **_: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
        shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
        residuals = {name: 0.1 for name in fids}
        stats = {name: None for name in fids}
        return shifts, residuals, stats

    monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
    monkeypatch.setattr(cli_register_module, "shift", lambda arr, *_a, **_k: arr)
    monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *_a, **_k: None)

    debug_messages: list[str] = []

    def fake_debug(message: object) -> None:
        debug_messages.append(str(message))

    monkeypatch.setattr(cli_register_module.logger, "debug", fake_debug)

    fids: dict[str, np.ndarray] = {
        "round_a": np.full((4, 4), 1, dtype=np.float32),
        "round_b": np.full((4, 4), 2, dtype=np.float32),
    }

    cli_register_module.run_fiducial(
        path=deconv,
        fids=fids,
        codebook_name="cb",
        config=cfg,
        roi="roiA",
        idx=0,
        reference="round_a",
        debug=True,
        fids_raw={k: v.copy() for k, v in fids.items()},
    )

    combined = "\n".join(debug_messages)
    assert "Applied priors:" in combined
    assert "round_b" in combined
    assert "dx=1.000" in combined
    assert "dy=2.000" in combined


def test_run_fiducial_debug_logs_when_no_priors(tmp_path: Path, monkeypatch: Any) -> None:
    _root, deconv = _make_workspace(tmp_path)

    cfg = Config(
        dataPath=str(DATA),
        exclude=None,
        registration=RegisterConfig(
            chromatic_path=DATA,
            fiducial=Fiducial(
                use_fft=False,
                fwhm=4.0,
                threshold=6.0,
                priors=None,
                overrides={},
                n_fids=2,
            ),
            reference="round_a",
            downsample=1,
            crop=0,
            slices=slice(None),
            reduce_bit_depth=0,
            discards=None,
        ),
    )

    def fake_align_with_stats(
        fids: dict[str, np.ndarray], **_: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
        shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
        residuals = {name: 0.1 for name in fids}
        stats = {name: None for name in fids}
        return shifts, residuals, stats

    monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
    monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *_a, **_k: None)

    debug_messages: list[str] = []

    def fake_debug(message: object) -> None:
        debug_messages.append(str(message))

    monkeypatch.setattr(cli_register_module.logger, "debug", fake_debug)

    fids: dict[str, np.ndarray] = {
        "round_a": np.full((4, 4), 1, dtype=np.float32),
        "round_b": np.full((4, 4), 2, dtype=np.float32),
    }

    cli_register_module.run_fiducial(
        path=deconv,
        fids=fids,
        codebook_name="cb",
        config=cfg,
        roi="roiA",
        idx=0,
        reference="round_a",
        debug=True,
        fids_raw={k: v.copy() for k, v in fids.items()},
    )

    combined = "\n".join(debug_messages)
    assert "Priors: none" in combined


def test_run_fiducial_auto_derives_priors_when_empty(tmp_path: Path, monkeypatch: Any) -> None:
    _root, deconv = _make_workspace(tmp_path)
    ws = cli_register_module.Workspace(deconv)

    roi = "roiA"
    codebook_name = "cb"

    shift_dir = ws.shifts(roi, codebook_name)
    shift_dir.mkdir(parents=True, exist_ok=True)
    payload = {"round_b": {"shifts": [1.0, 2.0], "corr": 1.0, "residual": 0.1}}
    for i in range(1, 12):  # >10 shift files triggers auto-priors
        (shift_dir / f"shifts-{i:04d}.json").write_text(json.dumps(payload))

    cfg = Config(
        dataPath=str(DATA),
        exclude=None,
        registration=RegisterConfig(
            chromatic_path=DATA,
            fiducial=Fiducial(
                use_fft=False,
                fwhm=4.0,
                threshold=6.0,
                priors={},  # important: empty dict should still auto-derive
                overrides={},
                n_fids=2,
            ),
            reference="round_a",
            downsample=1,
            crop=0,
            slices=slice(None),
            reduce_bit_depth=0,
            discards=None,
        ),
    )

    def fake_align_with_stats(
        fids: dict[str, np.ndarray], **_: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any]]:
        shifts = {name: np.array([0.0, 0.0], dtype=np.float32) for name in fids}
        residuals = {name: 0.1 for name in fids}
        stats = {name: None for name in fids}
        return shifts, residuals, stats

    monkeypatch.setattr(cli_register_module, "align_fiducials_with_stats", fake_align_with_stats)
    monkeypatch.setattr(cli_register_module, "safe_imwrite", lambda *_a, **_k: None)

    shift_calls: list[list[float]] = []

    def fake_shift(arr: np.ndarray, shift_vec: list[float], **_: Any) -> np.ndarray:
        shift_calls.append([float(shift_vec[0]), float(shift_vec[1])])
        return arr

    monkeypatch.setattr(cli_register_module, "shift", fake_shift)

    debug_messages: list[str] = []

    def fake_debug(message: object) -> None:
        debug_messages.append(str(message))

    monkeypatch.setattr(cli_register_module.logger, "debug", fake_debug)

    fids: dict[str, np.ndarray] = {
        "round_a": np.full((4, 4), 1, dtype=np.float32),
        "round_b": np.full((4, 4), 2, dtype=np.float32),
    }

    cli_register_module.run_fiducial(
        path=deconv,
        fids=fids,
        codebook_name=codebook_name,
        config=cfg,
        roi=roi,
        idx=0,
        reference="round_a",
        debug=True,
        fids_raw={k: v.copy() for k, v in fids.items()},
    )

    assert cfg.registration.fiducial.priors
    assert any(call == [2.0, 1.0] for call in shift_calls)  # dy, dx order in scipy.ndimage.shift
    assert "Applied priors:" in "\n".join(debug_messages)


def test_cli_register_batch_verify_respects_allow_large_shifts(tmp_path: Path, monkeypatch: Any) -> None:
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")

    cb = _make_codebook(tmp_path)

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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

        def registered(self, roi: str, codebook: str) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}"

        def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"shifts--{roi}+{codebook}" / f"shifts-{idx:04d}.json"

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        assert check is True
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
            "--threads",
            "1",
            "--verify",
            "--allow-large-shifts",
        ],
    )

    assert result.exit_code == 0, result.output
    # Missing outputs trigger verification reruns, so expect two calls with the flag.
    assert len(calls) == 2
    assert "--allow-large-shifts" in calls[0]
    assert "--allow-large-shifts" in calls[1]
    assert "--overwrite" in calls[1]


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
        repaired_rounds: set[str] | None = None,
        max_iters: int = 5,
        use_shifts_from: str | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        seen.update({"config": config, "roi": roi, "idx": idx, "reference": reference})

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(deconv),
            "7",
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
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


def test_cli_register_run_skips_when_reg_file_exists(tmp_path: Path, monkeypatch: Any) -> None:
    _root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    # Create the registered output file that triggers skip
    reg_dir = deconv / "registered--roiA+cb"
    reg_dir.mkdir(parents=True)
    (reg_dir / "reg-0042.tif").write_text("")

    called = False

    def fake__run(*_: Any, **__: Any) -> None:  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(deconv),
            "42",
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
            "--roi",
            "roiA",
            "--reference",
            "4_12_20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called is False


def test_run_internal_returns_early_when_reg_file_exists(tmp_path: Path) -> None:
    """Test that _run returns early when the output reg file already exists."""
    _root, deconv = _make_workspace(tmp_path)
    codebook_path = _make_codebook(tmp_path)
    reg_dir = deconv / "registered--roiA+cb"
    reg_dir.mkdir(parents=True)
    # Create the output file to trigger early return
    reg_file = reg_dir / "reg-0007.tif"
    reg_file.write_text("existing")

    cfg = Config(
        dataPath=str(DATA),
        exclude=None,
        registration=RegisterConfig(
            chromatic_path=DATA,
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
        path=deconv,
        roi="roiA",
        idx=7,
        codebook=codebook_path,
        reference="4_12_20",
        config=cfg,
        debug=False,
        overwrite=False,
        no_priors=False,
    )

    # The file should still have the original content (not overwritten)
    assert reg_file.read_text() == "existing"


def test_run_uses_previous_run_fids_when_reference_round_missing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _root, deconv = _make_workspace(tmp_path)
    roi = "roiA"
    reference = "2_10_18"
    target_round = "1_9_17"
    idx = 7

    # Only the target round exists; the reference round directory is missing.
    round_dir = deconv / f"{target_round}--{roi}"
    round_dir.mkdir(parents=True)
    (round_dir / f"{target_round}-{idx:04d}.tif").write_text("")

    # Previous run fiducials provide the reference fid plane.
    prev_fids_dir = deconv / f"registered--{roi}+cb" / "_fids"
    prev_fids_dir.mkdir(parents=True)
    ref_fid = np.arange(25, dtype=np.float32).reshape(5, 5)
    other_fid = np.zeros((5, 5), dtype=np.float32)
    imwrite(
        prev_fids_dir / f"_fids-{idx:04d}.tif",
        np.stack([ref_fid, other_fid]),
        metadata={"axes": "CYX", "key": [reference, target_round]},
    )

    class _StubImage:
        def __init__(self, name: str) -> None:
            self.name = name
            self.fid_raw = np.ones((5, 5), dtype=np.float32) * 7
            self.fid = np.ones((5, 5), dtype=np.float32) * 3

    def fake_from_file(path: Path, *_: Any, **__: Any) -> _StubImage:
        name, _idx = Path(path).stem.split("-")
        return _StubImage(name=name)

    def fake_run_fiducial(
        _path: Path,
        fids: dict[str, np.ndarray],
        _codebook_name: str,
        _config: Any,
        *,
        reference: str,
        fids_raw: dict[str, np.ndarray],
        **__: Any,
    ) -> dict[str, np.ndarray]:
        assert reference in fids
        np.testing.assert_array_equal(fids[reference], ref_fid)
        np.testing.assert_array_equal(fids_raw[reference], ref_fid)
        raise RuntimeError("stop-after-fid-load")

    monkeypatch.setattr("fishtools.preprocess.cli_register.Image.from_file", fake_from_file)
    monkeypatch.setattr("fishtools.preprocess.cli_register.run_fiducial", fake_run_fiducial)

    cb = tmp_path / "cb.json"
    cb.write_text('{"g":[1]}')

    cfg = Config(
        dataPath=str(DATA),
        exclude=None,
        registration=RegisterConfig(
            chromatic_path=DATA,
            fiducial=Fiducial(
                use_fft=True,
                fwhm=4.0,
                threshold=6.0,
                priors={},
                overrides={},
                n_fids=2,
            ),
            reference=reference,
            downsample=1,
            crop=40,
            slices=slice(None),
            reduce_bit_depth=0,
            discards=None,
        ),
    )

    with pytest.raises(RuntimeError, match="stop-after-fid-load"):
        _run(
            path=deconv,
            roi=roi,
            idx=idx,
            codebook=cb,
            reference=reference,
            config=cfg,
            debug=False,
            overwrite=True,
            no_priors=False,
        )


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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

        def registered(self, roi: str, codebook: str) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}"

        def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"shifts--{roi}+{codebook}" / f"shifts-{idx:04d}.json"

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], check: bool) -> Any:  # type: ignore[no-untyped-def]
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

        def registered(self, roi: str, codebook: str) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}"

        def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"shifts--{roi}+{codebook}" / f"shifts-{idx:04d}.json"

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

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
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

        @property
        def chromatic(self) -> Path:
            return self._deconved / "chromatic"

        @property
        def output(self) -> SimpleNamespace:
            output_root = self.path / "analysis" / "output"
            return SimpleNamespace(root=output_root, chromatic=output_root / "chromatic")

        def regimg(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}" / f"reg-{idx:04d}.tif"

        def registered(self, roi: str, codebook: str) -> Path:
            return self._deconved / f"registered--{roi}+{codebook}"

        def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
            return self._deconved / f"shifts--{roi}+{codebook}" / f"shifts-{idx:04d}.json"

    monkeypatch.setattr("fishtools.preprocess.cli_register.Workspace", _WS)

    # Child CLI should only be called for failing index despite overwrite not provided
    calls: list[list[str]] = []

    def fake_child(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_child)

    # TiffFile stub: 0001 always fails until fake_child "fixes" it; 0002 always OK
    # Track whether the child CLI has been called to fix the file
    fixed_files: set[str] = set()
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
            # 0001 fails until the child CLI "fixes" it
            if name == "reg-0001.tif" and name not in fixed_files:
                raise OSError("simulated read error")
            return np.zeros(baseline_shape, dtype=np.uint16)

    # Update fake_child to mark files as fixed
    original_fake_child = fake_child

    def fake_child_with_fix(argv: list[str], *, check: bool = True):  # type: ignore[no-untyped-def]
        result = original_fake_child(argv, check=check)
        # Mark the file as fixed after child CLI runs
        idx = int(argv[4])
        fixed_files.add(f"reg-{idx:04d}.tif")
        return result

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_child_with_fix)

    monkeypatch.setattr("fishtools.preprocess.cli_register.TiffFile", _TF)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": "2_10_18"})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
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


def _make_shift_json(shifts: dict[str, tuple[float, float]]) -> str:
    """Create shift JSON in the format used by Shifts TypeAdapter."""
    data = {
        name: {"shifts": list(s), "corr": 0.99, "residual": 0.1}
        for name, s in shifts.items()
    }
    return Shifts.dump_json(Shifts.validate_python(data)).decode()


def test_load_shifts_from_codebook_reads_existing_shifts(tmp_path: Path) -> None:
    """Test that _load_shifts_from_codebook loads shifts from source codebook."""
    from fishtools.io.workspace import Workspace

    root, deconv = _make_workspace(tmp_path)
    ws = Workspace(deconv)
    roi = "roiA"
    source_cb = "source_cb"
    idx = 7

    # Create shift file for source codebook
    shift_dir = deconv / f"shifts--{roi}+{source_cb}"
    shift_dir.mkdir(parents=True)
    expected_shifts = {"2_10_18": (1.5, -2.3), "1_9_17": (0.5, 0.7)}
    (shift_dir / f"shifts-{idx:04d}.json").write_text(_make_shift_json(expected_shifts))

    result = _load_shifts_from_codebook(ws, roi=roi, source_codebook=source_cb, idx=idx)

    assert set(result.keys()) == {"2_10_18", "1_9_17"}
    np.testing.assert_array_almost_equal(result["2_10_18"], [1.5, -2.3])
    np.testing.assert_array_almost_equal(result["1_9_17"], [0.5, 0.7])


def test_load_shifts_from_codebook_raises_when_missing(tmp_path: Path) -> None:
    """Test that _load_shifts_from_codebook raises FileNotFoundError when shift file missing."""
    from fishtools.io.workspace import Workspace

    _root, deconv = _make_workspace(tmp_path)
    ws = Workspace(deconv)

    with pytest.raises(FileNotFoundError, match="Shift file not found"):
        _load_shifts_from_codebook(ws, roi="roiA", source_codebook="nonexistent", idx=1)


def test_cli_register_run_use_shifts_from_skips_fiducial_registration(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Test that --use-shifts-from loads shifts and skips fiducial registration."""
    _root, deconv = _make_workspace(tmp_path)
    cb = _make_codebook(tmp_path)

    # Create source shifts
    source_cb = "source_cb"
    shift_dir = deconv / f"shifts--roiA+{source_cb}"
    shift_dir.mkdir(parents=True)
    (shift_dir / "shifts-0042.json").write_text(
        _make_shift_json({"2_10_18": (1.0, 2.0), "1_9_17": (0.5, 0.5)})
    )

    called: dict[str, Any] = {}
    fiducial_called = False

    def fake__run(
        path: Path,
        roi: str,
        idx: int,
        *,
        codebook: str | Path,
        reference: str,
        config: Any,
        debug: bool,
        overwrite: bool,
        no_priors: bool,
        repaired_rounds: set[str] | None = None,
        max_iters: int = 5,
        use_shifts_from: str | None = None,
    ) -> None:
        called.update({
            "path": path,
            "roi": roi,
            "idx": idx,
            "use_shifts_from": use_shifts_from,
        })

    def fake_run_fiducial(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        nonlocal fiducial_called
        fiducial_called = True
        return {}

    monkeypatch.setattr("fishtools.preprocess.cli_register._run", fake__run)
    monkeypatch.setattr("fishtools.preprocess.cli_register.run_fiducial", fake_run_fiducial)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "run",
            str(deconv),
            "42",
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
            "--roi",
            "roiA",
            "--reference",
            "2_10_18",
            f"--use-shifts-from={source_cb}",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["use_shifts_from"] == source_cb


def test_cli_register_batch_forwards_use_shifts_from(tmp_path: Path, monkeypatch: Any) -> None:
    """Test that batch command forwards --use-shifts-from to child CLI calls."""
    _root, base = _make_workspace(tmp_path)
    (base / "2_10_18--roiA").mkdir(parents=True)
    (base / "2_10_18--roiA" / "2_10_18-0001.tif").write_text("")
    (base / "shifts--roiA+other_codebook").mkdir(parents=True)
    (base / "shifts--roiA+other_codebook" / "shifts-0001.json").write_text("{}")

    cb = _make_codebook(tmp_path)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True) -> Any:
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA)

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(base),
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
            "--threads",
            "1",
            "--overwrite",
            "--use-shifts-from=other_codebook",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert "--use-shifts-from=other_codebook" in calls[0]


def test_cli_register_batch_use_shifts_from_does_not_require_ref_images(tmp_path: Path, monkeypatch: Any) -> None:
    """When --use-shifts-from is set, batch should discover indices from shift files, not ref images."""
    _root, deconv = _make_workspace(tmp_path)
    roi = "roiA"
    ref = "2_10_18"
    source_cb = "source_cb"

    # Ensure ROI exists but do NOT create any ref round images.
    (deconv / f"1_9_17--{roi}").mkdir(parents=True)

    # Create shift files that batch can use to enumerate indices.
    shift_dir = deconv / f"shifts--{roi}+{source_cb}"
    shift_dir.mkdir(parents=True)
    (shift_dir / "shifts-0001.json").write_text("{}")
    (shift_dir / "shifts-0002.json").write_text("{}")

    cb = _make_codebook(tmp_path)

    calls: list[list[str]] = []

    def fake_run(argv: list[str], *, check: bool = True) -> Any:
        calls.append(argv)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("fishtools.preprocess.cli_register._run_child_cli", fake_run)

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": ref})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "batch",
            str(deconv),
            "--reference",
            ref,
            "--codebook",
            str(cb),
            "--config",
            str(config_path),
            "--threads",
            "1",
            f"--use-shifts-from={source_cb}",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert {argv[4] for argv in calls} == {"1", "2"}


def test_run_internal_uses_shifts_from_source_codebook(tmp_path: Path, monkeypatch: Any) -> None:
    """Test that _run loads shifts from source codebook and skips fiducial registration."""
    _root, deconv = _make_workspace(tmp_path)
    roi = "roiA"
    target_round = "1_9_17"
    reference = "2_10_18"
    idx = 7
    source_cb = "source_cb"

    # Create target round directory with image
    round_dir = deconv / f"{target_round}--{roi}"
    round_dir.mkdir(parents=True)
    (round_dir / f"{target_round}-{idx:04d}.tif").write_text("")

    # Create source shifts
    shift_dir = deconv / f"shifts--{roi}+{source_cb}"
    shift_dir.mkdir(parents=True)
    expected_shifts = {reference: (0.0, 0.0), target_round: (3.5, -1.2)}
    (shift_dir / f"shifts-{idx:04d}.json").write_text(_make_shift_json(expected_shifts))

    # Track whether run_fiducial is called (it shouldn't be)
    fiducial_called = False

    def fake_run_fiducial(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        nonlocal fiducial_called
        fiducial_called = True
        return {}

    # Stub Image.from_file to avoid actual file I/O
    class _StubImage:
        def __init__(self, name: str) -> None:
            self.name = name
            self.fid_raw = np.ones((5, 5), dtype=np.float32)
            self.fid = np.ones((5, 5), dtype=np.float32)
            self.nofid = np.ones((1, 1, 5, 5), dtype=np.float32)
            self.bits = [name.split("_")[0]]
            self.powers = {"560": 1.0}
            self.metadata = {"prenormalized": True}
            self.global_deconv_scaling = None
            self.basic = lambda: None

    def fake_from_file(path: Path, *_: Any, **__: Any) -> _StubImage:
        name, _idx = Path(path).stem.split("-")
        return _StubImage(name=name)

    monkeypatch.setattr("fishtools.preprocess.cli_register.Image.from_file", fake_from_file)
    monkeypatch.setattr("fishtools.preprocess.cli_register.run_fiducial", fake_run_fiducial)

    cb = tmp_path / "cb.json"
    cb.write_text('{"g":["1"]}')

    cfg = Config(
        dataPath=str(DATA),
        exclude=None,
        registration=RegisterConfig(
            chromatic_path=DATA,
            fiducial=Fiducial(
                use_fft=True,
                fwhm=4.0,
                threshold=6.0,
                priors={},
                overrides={},
                n_fids=2,
            ),
            reference=reference,
            downsample=1,
            crop=40,
            slices=slice(None),
            reduce_bit_depth=0,
            discards=None,
        ),
    )

    # The _run should fail later (no actual images), but fiducial should be skipped
    with pytest.raises(Exception):  # Will fail when trying to process images
        _run(
            path=deconv,
            roi=roi,
            idx=idx,
            codebook=cb,
            reference=reference,
            config=cfg,
            debug=False,
            overwrite=True,
            no_priors=False,
            use_shifts_from=source_cb,
        )

    # The key assertion: run_fiducial should NOT have been called
    assert fiducial_called is False


def test_cli_fix_shifts_accepts_star_roi(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "ws"
    deconv = root / "analysis" / "deconv"
    deconv.mkdir(parents=True)
    (root / "workspace.DONE").write_text("")

    reference = "2_10_18"
    target_round = "1_9_17"
    rois = ["roiA", "roiB"]

    for roi in rois:
        for round_name in [reference, target_round]:
            round_dir = root / f"{round_name}--{roi}"
            round_dir.mkdir(parents=True)
            imwrite(round_dir / f"{round_name}-0001.tif", np.zeros((1, 1), dtype=np.uint16))

    class _Image:
        def __init__(self) -> None:
            self.fid = np.zeros((8, 8), dtype=np.float32)
            self.fid_raw = np.zeros((8, 8), dtype=np.float32)

        @classmethod
        def from_file(cls, _: Path, *, n_fids: int = 2) -> "_Image":
            return cls()

    def fake_align(fids: dict[str, np.ndarray], *, reference: str, **_: Any):
        shifts = {name: np.array([0.0, 0.0]) for name in fids}
        residuals = {name: 0.0 for name in fids}
        if reference in shifts:
            shifts[reference] = np.array([0.0, 0.0])
            residuals[reference] = 0.0
        return shifts, residuals, {}

    monkeypatch.setattr("fishtools.preprocess.cli_register.Image", _Image)
    monkeypatch.setattr("fishtools.preprocess.cli_register.align_fiducials_with_stats", fake_align)
    monkeypatch.setattr("fishtools.preprocess.cli_register._silence_matplotlib_debug_logs", lambda: None)
    monkeypatch.setattr(
        "fishtools.preprocess.cli_register.setup_cli_logging",
        lambda *_, **__: Path("dummy.log"),
    )

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": reference})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "fix-shifts",
            str(root),
            "1",
            "--config",
            str(config_path),
            "--roi",
            "*",
            "--rounds",
            target_round,
            "--reference",
            reference,
        ],
    )

    assert result.exit_code == 0, result.output
    for roi in rois:
        output_path = root / "analysis" / "deconv" / f"shifts--{roi}" / "coarse_shifts.json"
        assert output_path.exists()


def test_cli_fix_shifts_merges_existing_tiles(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "ws"
    deconv = root / "analysis" / "deconv"
    shifts_dir = deconv / "shifts--roiA"
    shifts_dir.mkdir(parents=True)
    (root / "workspace.DONE").write_text("")

    reference = "2_10_18"
    target_round = "1_9_17"
    roi = "roiA"

    for round_name in [reference, target_round]:
        round_dir = root / f"{round_name}--{roi}"
        round_dir.mkdir(parents=True)
        imwrite(round_dir / f"{round_name}-0001.tif", np.zeros((1, 1), dtype=np.uint16))
        imwrite(round_dir / f"{round_name}-0002.tif", np.zeros((1, 1), dtype=np.uint16))

    existing = {
        "reference": "old_ref",
        "use_fft": False,
        "tiles": {
            "0002": {
                target_round: {"dx": 1.0, "dy": 2.0, "magnitude": 2.2, "residual": 0.1}
            }
        },
    }
    (shifts_dir / "coarse_shifts.json").write_text(json.dumps(existing))

    class _Image:
        def __init__(self) -> None:
            self.fid = np.zeros((8, 8), dtype=np.float32)
            self.fid_raw = np.zeros((8, 8), dtype=np.float32)

        @classmethod
        def from_file(cls, _: Path, *, n_fids: int = 2) -> "_Image":
            return cls()

    def fake_align(fids: dict[str, np.ndarray], *, reference: str, **_: Any):
        shifts = {name: np.array([0.0, 0.0]) for name in fids}
        residuals = {name: 0.0 for name in fids}
        if reference in shifts:
            shifts[reference] = np.array([0.0, 0.0])
            residuals[reference] = 0.0
        return shifts, residuals, {}

    monkeypatch.setattr("fishtools.preprocess.cli_register.Image", _Image)
    monkeypatch.setattr("fishtools.preprocess.cli_register.align_fiducials_with_stats", fake_align)
    monkeypatch.setattr("fishtools.preprocess.cli_register._silence_matplotlib_debug_logs", lambda: None)
    monkeypatch.setattr(
        "fishtools.preprocess.cli_register.setup_cli_logging",
        lambda *_, **__: Path("dummy.log"),
    )

    config_path = _write_config_file(tmp_path, chromatic_path=DATA, registration={"reference": reference})

    runner = CliRunner()
    result = runner.invoke(
        register_cli,
        [
            "fix-shifts",
            str(root),
            "1",
            "--config",
            str(config_path),
            "--roi",
            roi,
            "--rounds",
            target_round,
            "--reference",
            reference,
        ],
    )

    assert result.exit_code == 0, result.output
    output_path = shifts_dir / "coarse_shifts.json"
    output_data = json.loads(output_path.read_text())
    assert "0001" in output_data["tiles"]
    assert output_data["tiles"]["0002"][target_round]["dx"] == 1.0
