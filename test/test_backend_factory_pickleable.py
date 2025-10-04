import pickle
from pathlib import Path

import numpy as np

from fishtools.preprocess.cli_deconv import _BACKEND_CLASSES
from fishtools.preprocess.deconv.backend import ProcessorConfig, make_processor_factory
from fishtools.preprocess.config import DeconvolutionOutputMode


def _dummy_config(tmp_path: Path) -> ProcessorConfig:
    # Minimal, no IO performed when only pickling the factory.
    return ProcessorConfig(
        round_name="r1",
        basic_paths=[tmp_path / "basic" / "all-000.pkl"],
        output_dir=tmp_path / "analysis" / "deconv",
        n_fids=0,
        step=6,
        mode=DeconvolutionOutputMode.F32,
        histogram_bins=16,
        m_glob=None,
        s_glob=None,
        debug=False,
    )


def test_build_backend_factory_float32_pickles(tmp_path: Path):
    factory = _BACKEND_CLASSES[DeconvolutionOutputMode.F32]
    # Top-level function should pickle fine.
    pickle.dumps(factory)

    cfg = _dummy_config(tmp_path)
    pf = make_processor_factory(cfg, backend_factory=factory)
    pickle.dumps(pf)


def test_build_backend_factory_u16_pickles(tmp_path: Path):
    factory = _BACKEND_CLASSES[DeconvolutionOutputMode.U16]
    pickle.dumps(factory)

    cfg = _dummy_config(tmp_path)
    cfg = cfg.__class__(
        round_name=cfg.round_name,
        basic_paths=cfg.basic_paths,
        output_dir=cfg.output_dir,
        n_fids=cfg.n_fids,
        step=cfg.step,
        mode=DeconvolutionOutputMode.U16,
        histogram_bins=cfg.histogram_bins,
        m_glob=np.array([[0.0]], dtype=np.float32),
        s_glob=np.array([[1.0]], dtype=np.float32),
        debug=cfg.debug,
    )
    pf = make_processor_factory(cfg, backend_factory=factory)
    pickle.dumps(pf)
