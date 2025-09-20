from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner

from fishtools.postprocess.subtraction import app

runner = CliRunner()


def test_fit_cli_raw(raw_bleed_setup: tuple[Path, Path, float, float]) -> None:
    analysis_path, preprocessed_dir, slope_true, intercept_true = raw_bleed_setup

    # Fit regression coefficients from precomputed .npz slices
    result_fit = runner.invoke(
        app,
        [
            "fit",
            str(analysis_path),
            str(preprocessed_dir),
            "--output-filename",
            "bleed.csv",
            "--fit-threshold",
            "0",
            "--num-bins",
            "10",
            "--min-bin-count",
            "10",
        ],
    )
    assert result_fit.exit_code == 0, result_fit.output

    output_csv = analysis_path / "bleed.csv"
    assert output_csv.exists(), result_fit.output
    df = pd.read_csv(output_csv)
    assert not df.empty
    params = df.iloc[0]
    assert np.isfinite(params["slope"])
    assert np.isfinite(params["intercept"])
    # Expect slope reasonably close to the injected ground truth
    assert params["slope"] == pytest.approx(slope_true, rel=0.35)
    assert params["intercept"] == pytest.approx(intercept_true, rel=0.6)


def test_fit_and_subtract_registered_cli(
    registered_stack_data: tuple[Path, np.ndarray, dict[str, int], float, float, np.ndarray, np.ndarray],
) -> None:
    stack_path, stack_uint16, channel_index, slope_true, intercept_true, signal_float, blank_float = (
        registered_stack_data
    )
    params_csv = stack_path.with_name("registered_params.csv")
    output_path = stack_path.with_name("registered_corrected.tif")

    # Fit directly on the registered stack using channel names
    result_fit = runner.invoke(
        app,
        [
            "fit-registered",
            str(stack_path),
            str(params_csv),
            "--signal-channel",
            "geneA",
            "--blank-channel",
            "Blank-560",
            "--fit-threshold",
            "0",
            "--num-bins",
            "10",
            "--min-bin-count",
            "10",
        ],
    )
    assert result_fit.exit_code == 0, result_fit.output

    assert params_csv.exists(), result_fit.output
    df = pd.read_csv(params_csv)
    params = df.iloc[0]
    assert params["signal_channel"] == "geneA"
    assert params["blank_channel"] == "Blank-560"
    assert params["slope"] == pytest.approx(slope_true, rel=0.05)
    assert params["intercept"] == pytest.approx(intercept_true, rel=0.2)

    # Apply subtraction using the fitted coefficients
    result_sub = runner.invoke(
        app,
        [
            "subtract-registered",
            str(stack_path),
            str(output_path),
            "--signal-channel",
            "geneA",
            "--blank-channel",
            "Blank-560",
            "--params-csv",
            str(params_csv),
        ],
    )
    assert result_sub.exit_code == 0, result_sub.output

    with tifffile.TiffFile(output_path) as tif:
        corrected = tif.asarray()
        meta = tif.shaped_metadata[0]  # type: ignore

    assert meta.get("subtract", {}).get("signal_channel") == "geneA"
    assert meta.get("subtract", {}).get("blank_channel") == "Blank-560"

    corrected_signal = corrected[:, channel_index["geneA"]].astype(np.float32)

    expected = signal_float - (blank_float * params["slope"] + params["intercept"])
    assert np.mean(expected) >= -50  # sanity check baseline
    expected_clipped = np.clip(expected, 0, np.iinfo(stack_uint16.dtype).max)
    assert np.allclose(corrected_signal, expected_clipped, atol=6)


def test_subtract_registered_manual_params(
    registered_stack_data: tuple[Path, np.ndarray, dict[str, int], float, float, np.ndarray, np.ndarray],
) -> None:
    stack_path, stack_uint16, channel_index, slope_true, intercept_true, signal_float, blank_float = (
        registered_stack_data
    )
    manual_out = stack_path.with_name("manual_corrected.tif")

    result_sub = runner.invoke(
        app,
        [
            "subtract-registered",
            str(stack_path),
            str(manual_out),
            "--signal-channel",
            "geneA",
            "--blank-channel",
            "Blank-560",
            "--slope",
            f"{slope_true}",
            "--intercept",
            f"{intercept_true}",
        ],
    )
    assert result_sub.exit_code == 0, result_sub.output

    with tifffile.TiffFile(manual_out) as tif:
        corrected = tif.asarray()

    corrected_signal = corrected[:, channel_index["geneA"]].astype(np.float32)
    expected = signal_float - (blank_float * slope_true + intercept_true)
    expected_clipped = np.clip(expected, 0, np.iinfo(stack_uint16.dtype).max)
    assert np.allclose(corrected_signal, expected_clipped, atol=6)
