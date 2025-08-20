from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from fishtools.preprocess.tileconfig import TileConfiguration


@pytest.fixture
def sample_df_pl() -> pl.DataFrame:
    data = {
        "index": [1, 2, 3],
        "filename": ["0001.tif", "0002.tif", "img-0003.tif"],
        "x": [0.0, 100.0, 200.0],
        "y": [0.0, 50.0, 150.0],
        "prefix": [None, None, "img-"],
    }
    return pl.DataFrame(data).with_columns(
        pl.col("index").cast(pl.Int64), pl.col("x").cast(pl.Float64), pl.col("y").cast(pl.Float64)
    )


@pytest.fixture
def sample_tile_config(sample_df_pl: pl.DataFrame) -> TileConfiguration:
    # Keep only relevant columns for TileConfiguration internal df
    df_for_tc = sample_df_pl.select(["index", "filename", "x", "y"])
    return TileConfiguration(df_for_tc)


def test_tile_configuration_init_and_len(sample_df_pl: pl.DataFrame):
    df_for_tc = sample_df_pl.select(["index", "filename", "x", "y"])
    tc = TileConfiguration(df_for_tc)
    assert len(tc) == 3
    assert_frame_equal(tc.df, df_for_tc)


def test_write_and_from_file(sample_tile_config: TileConfiguration, tmp_path: Path):
    file_path = tmp_path / "TileConfiguration.txt"

    # Modify sample_tile_config to match expected write format (no prefix, specific filename format)
    # The write method generates filenames like "0001.tif" from "index"
    # It also expects x and y to be floats for the string formatting.
    df_for_write = sample_tile_config.df.with_columns(
        pl.col("x").cast(pl.Float64), pl.col("y").cast(pl.Float64)
    )
    # The `write` method only uses 'index', 'x', 'y'. 'filename' is reconstructed.
    # So, the `from_file` will parse the reconstructed filename.

    # Create a TC instance with data that write method can handle properly
    # and from_file can parse back.
    original_data_for_write = pl.DataFrame(
        {
            "index": [1, 2, 3],
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 50.0, 150.0],
        },
        schema=pl.Schema({"index": pl.UInt32, "x": pl.Float32, "y": pl.Float32}),
    )
    tc_to_write = TileConfiguration(original_data_for_write)
    tc_to_write.write(file_path)

    assert file_path.exists()

    loaded_tc = TileConfiguration.from_file(file_path)

    # from_file adds 'filename' and 'prefix' (which will be None here)
    expected_df_after_load = original_data_for_write.with_columns(
        filename=(pl.col("index").cast(pl.Utf8).str.zfill(4) + ".tif"),
        prefix=pl.lit(None, dtype=pl.Utf8),
    ).select(["prefix", "index", "filename", "x", "y"])  # Ensure column order

    assert len(loaded_tc) == len(tc_to_write)
    # Sort by index before comparing as order might not be guaranteed by parser if not already sorted
    assert_frame_equal(loaded_tc.df.sort("index"), expected_df_after_load.sort("index"))


def test_from_file_parsing_logic(tmp_path: Path):
    content = """dim=2
# This is a comment
0001.tif; ; (0.0, 0.0)
CH1-0002.tif; ; (10.5, -20.0)
0003.tif; ; (30.0, 40.12345)
# Another comment
"""
    file_path = tmp_path / "TileConfiguration_complex.txt"
    file_path.write_text(content)

    tc = TileConfiguration.from_file(file_path)

    expected_data = {
        "prefix": [None, "CH1-", None],
        "index": [1, 2, 3],
        "filename": ["0001.tif", "CH1-0002.tif", "0003.tif"],
        "x": [0.0, 10.5, 30.0],
        "y": [0.0, -20.0, 40.12345],
    }
    expected_df = pl.DataFrame(
        expected_data,
        schema=pl.Schema(
            {
                "prefix": pl.Utf8,
                "index": pl.UInt32,
                "filename": pl.Utf8,
                "x": pl.Float32,
                "y": pl.Float32,
            }
        ),
    )

    assert len(tc) == 3
    assert_frame_equal(tc.df.sort("index"), expected_df.sort("index"))


def test_from_pos():
    # pixel = 2048
    # actual_per_pixel = 0.108
    # microscope_scaling_factor = 200
    # final_scaling = -(1 / actual_per_pixel)

    # Input data: raw stage positions (e.g., in microns)
    # (df[0] is Y, df[1] is X based on TileConfiguration.from_pos implementation)
    raw_pos_data = {
        0: [1000.0, 1000.0 + 2048 * 0.108, 1000.0 + 2 * 2048 * 0.108],  # Stage Y
        1: [500.0, 500.0 + 2048 * 0.108, 500.0 - 2048 * 0.108],  # Stage X
    }
    df_pos_pd = pd.DataFrame(raw_pos_data)

    tc = TileConfiguration.from_pos(df_pos_pd)

    # Expected logic:
    # 1. y_adj = (df[0] - df[0].min())
    # 2. x_adj = (df[1] - df[1].min())
    # 3. x_adj_scaled = (x_adj - x_adj.min()) * -(1/0.108)
    # 4. y_adj_scaled = (y_adj - y_adj.min()) * -(1/0.108)
    # 5. ats_x = x_adj_scaled - x_adj_scaled.min()
    # 6. ats_y = y_adj_scaled - y_adj_scaled.min()

    y_min = df_pos_pd[0].min()
    x_min = df_pos_pd[1].min()

    y_adj = df_pos_pd[0] - y_min
    x_adj = df_pos_pd[1] - x_min

    # Scaling part in from_pos:
    # adjusted["x"] *= -(1 / 200) * pixel * scaling
    # adjusted["y"] *= -(1 / 200) * pixel * scaling
    # where pixel = 2048, scaling = 200 / (pixel * 0.108)
    # So, factor = -(1/200) * pixel * (200 / (pixel * 0.108)) = -1 / 0.108
    scaling_factor = -1 / 0.108

    x_scaled = (x_adj - x_adj.min()) * scaling_factor
    y_scaled = (y_adj - y_adj.min()) * scaling_factor

    # Final shift to ensure min is 0
    expected_x = x_scaled - x_scaled.min()
    expected_y = y_scaled - y_scaled.min()

    expected_df = pl.DataFrame(
        {
            "index": [0, 1, 2],  # from reset_index()
            "y": list(expected_y),
            "x": list(expected_x),
        },
        schema=pl.Schema({"index": pl.UInt32, "x": pl.Float32, "y": pl.Float32}),
    ).select(["index", "x", "y"])  # Ensure column order and types

    assert len(tc) == 3
    assert "x" in tc.df.columns
    assert "y" in tc.df.columns
    assert "index" in tc.df.columns

    # Check that min x and y are close to 0
    assert tc.df["x"].min() == pytest.approx(0.0)
    assert tc.df["y"].min() == pytest.approx(0.0)

    # Compare the calculated values
    assert_frame_equal(tc.df.select(["index", "x", "y"]), expected_df, atol=1e-6)


def test_drop(sample_tile_config: TileConfiguration):
    original_len = len(sample_tile_config)
    indices_to_drop = [2]  # Corresponds to index value 2

    tc_dropped = sample_tile_config.drop(indices_to_drop)

    assert len(tc_dropped) == original_len - len(indices_to_drop)
    assert 2 not in tc_dropped.df["index"].to_list()
    assert 1 in tc_dropped.df["index"].to_list()
    assert 3 in tc_dropped.df["index"].to_list()  # This should be dropped

    # Corrected assertion for dropped items
    indices_to_drop_val = [2]
    tc_dropped_correct = sample_tile_config.drop(indices_to_drop_val)
    assert len(tc_dropped_correct) == original_len - len(indices_to_drop_val)
    assert 2 not in tc_dropped_correct.df["index"].to_list()
    assert 1 in tc_dropped_correct.df["index"].to_list()
    # Index 3 was "img-0003.tif" with index 3 in sample_df_pl
    # So if we drop index 2, index 1 and 3 should remain.
    assert 3 in tc_dropped_correct.df["index"].to_list()

    # Test dropping multiple
    indices_to_drop_multi = [1, 3]
    tc_dropped_multi = sample_tile_config.drop(indices_to_drop_multi)
    assert len(tc_dropped_multi) == original_len - len(indices_to_drop_multi)
    assert 1 not in tc_dropped_multi.df["index"].to_list()
    assert 3 not in tc_dropped_multi.df["index"].to_list()
    assert 2 in tc_dropped_multi.df["index"].to_list()


def test_downsample(sample_tile_config: TileConfiguration):
    factor = 2
    tc_downsampled = sample_tile_config.downsample(factor)

    expected_x = sample_tile_config.df["x"] / factor
    expected_y = sample_tile_config.df["y"] / factor

    assert_series_equal(tc_downsampled.df["x"], expected_x)
    assert_series_equal(tc_downsampled.df["y"], expected_y)
    assert len(tc_downsampled) == len(sample_tile_config)

    # Test with factor 1
    tc_downsampled_one = sample_tile_config.downsample(1)
    assert_frame_equal(tc_downsampled_one.df, sample_tile_config.df)


def test_getitem_slice(sample_tile_config: TileConfiguration):
    sl = slice(0, 2)
    tc_sliced = sample_tile_config[sl]

    assert len(tc_sliced) == 2
    assert_frame_equal(tc_sliced.df, sample_tile_config.df[sl])


def test_from_file_empty_file(tmp_path: Path):
    file_path = tmp_path / "empty_TileConfiguration.txt"
    file_path.write_text("")
    with pytest.raises(ValueError, match="is empty or not in the expected format."):
        TileConfiguration.from_file(file_path)


def test_from_file_only_comments_or_dim(tmp_path: Path):
    content = """dim=2
# comment 1
# comment 2
dim=3 # another dim line
"""
    file_path = tmp_path / "comments_TileConfiguration.txt"
    file_path.write_text(content)
    with pytest.raises(ValueError, match="empty or not in the expected format."):
        TileConfiguration.from_file(file_path)
