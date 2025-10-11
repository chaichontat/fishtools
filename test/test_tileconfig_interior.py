import numpy as np
import polars as pl
import pytest

from fishtools.preprocess.tileconfig import (
    TileConfiguration,
    tiles_at_least_n_steps_from_edges,
    interior_indices_geometry,
)


def _grid_xy(nx: int, ny: int, *, xstep: float = 1.0, ystep: float = 1.0) -> np.ndarray:
    xs = np.arange(nx, dtype=float) * xstep
    ys = np.arange(ny, dtype=float) * ystep
    xy = [(x, y) for y in ys for x in xs]
    return np.asarray(xy, dtype=float)


def test_interior_indices_square_grid_n1() -> None:
    """5x4 grid, n=1 → drop outer ring, keep 3x2 center."""
    nx, ny, n = 5, 4, 1
    xy = _grid_xy(nx, ny)
    pos = tiles_at_least_n_steps_from_edges(xy, n)
    # Expected positions in row-major order: y in {1,2}, x in {1,2,3}
    expected = np.array([5 + 1, 5 + 2, 5 + 3, 10 + 1, 10 + 2, 10 + 3])
    assert np.array_equal(np.sort(pos), np.sort(expected))


def test_interior_indices_irregular_spacing() -> None:
    """Irregular x/y spacing should not change ordinal selection."""
    nx, ny, n = 5, 4, 1
    xs = np.array([0.0, 0.7, 1.4, 3.9, 8.2])
    ys = np.array([-2.0, 0.0, 0.1, 10.0])
    xy = np.asarray([(x, y) for y in ys for x in xs], dtype=float)
    pos = tiles_at_least_n_steps_from_edges(xy, n)
    expected = np.array([5 + 1, 5 + 2, 5 + 3, 10 + 1, 10 + 2, 10 + 3])
    assert np.array_equal(np.sort(pos), np.sort(expected))


def test_geometry_uniform_grid_matches_ordinal() -> None:
    """On a uniform 5x4 grid, geometry-based selection equals ordinal for n=1."""
    xy = _grid_xy(5, 4)
    pos_ord = tiles_at_least_n_steps_from_edges(xy, 1)
    pos_geo = interior_indices_geometry(xy, 1)
    assert np.array_equal(np.sort(pos_geo), np.sort(pos_ord))


def test_interior_indices_center_only_n1_on_3x3() -> None:
    """3x3 grid, n=1 → only center tile remains (index 4)."""
    xy = _grid_xy(3, 3)
    pos = tiles_at_least_n_steps_from_edges(xy, 1)
    assert np.array_equal(pos, np.array([4]))


def test_interior_indices_empty_when_not_enough_layers() -> None:
    """3x3 grid, n=2 → no tile is ≥2 steps away on both axes."""
    xy = _grid_xy(3, 3)
    pos = tiles_at_least_n_steps_from_edges(xy, 2)
    assert pos.size == 0


def test_interior_indices_single_row_returns_empty() -> None:
    """A 10x1 grid has no interior for n>=1."""
    xy = _grid_xy(10, 1)
    pos = tiles_at_least_n_steps_from_edges(xy, 1)
    assert pos.size == 0


def test_interior_indices_with_duplicates_in_x() -> None:
    """Duplicate x values: all tiles at interior x-ranks should be kept."""
    # Unique x values 0,1,2,3 with x=2 duplicated once per row
    xs_unique = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 1.0, 2.0, 3.0])
    xs = np.concatenate([xs_unique[:3], [2.0], xs_unique[3:]])  # 0,1,2,2,3 → len=5 per row
    xy = np.asarray([(x, y) for y in ys for x in xs], dtype=float)
    pos = tiles_at_least_n_steps_from_edges(xy, 1)
    # For each of ny=4 rows, interior x-ranks are 1..2 (unique ranks), and x=2.0 appears twice.
    # Thus per interior row (y=1,2) we expect indices at columns {1,2,3} → 3 per row.
    row_len = len(xs)
    expected = np.array(
        [
            row_len * 1 + 1,
            row_len * 1 + 2,
            row_len * 1 + 3,
            row_len * 2 + 1,
            row_len * 2 + 2,
            row_len * 2 + 3,
        ]
    )
    assert np.array_equal(np.sort(pos), np.sort(expected))


def test_interior_indices_shuffled_input_positions() -> None:
    """Order of input should not affect which positions are selected."""
    rng = np.random.default_rng(0)
    xy = _grid_xy(5, 4)
    pos_ref = tiles_at_least_n_steps_from_edges(xy, 1)
    perm = rng.permutation(len(xy))
    pos_shuffled = tiles_at_least_n_steps_from_edges(xy[perm], 1)
    # Map back to original positions using the permutation inverse
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    assert np.array_equal(np.sort(perm[pos_shuffled]), np.sort(pos_ref))


def test_tileconfiguration_indices_map_to_index_column() -> None:
    """TileConfiguration method must return IDs from the 'index' column, not row positions."""
    nx, ny = 4, 4
    xy = _grid_xy(nx, ny)
    # Construct a dataframe with nontrivial indices (reverse order numbering)
    # and shuffled rows to ensure mapping is by content, not position.
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(xy))
    df = pl.DataFrame(
        {
            "index": pl.Series((len(xy) - 1 - np.arange(len(xy))).astype(np.uint32)),
            "x": pl.Series(xy[perm, 0].astype(np.float32)),
            "y": pl.Series(xy[perm, 1].astype(np.float32)),
        }
    )
    tc = TileConfiguration(df)
    ids = tc.indices_at_least_n_steps_from_edges(1)
    # Expected: compute over the same (permuted) XY ordering as the dataframe, then map to the
    # reversed IDs in that order.
    pos_perm = tiles_at_least_n_steps_from_edges(xy[perm], 1)
    reversed_ids = (len(xy) - 1 - np.arange(len(xy))).astype(np.uint32)
    expected_ids = reversed_ids[pos_perm]
    assert np.array_equal(np.sort(ids), np.sort(expected_ids))


def test_geometry_concave_outline_excludes_notch_adjacent() -> None:
    """Geometry method drops tiles adjacent to a concave notch (hole)."""
    # 5x5 grid, remove a 2x2 inner block to create a central hole at (2,2),(2,3),(3,2),(3,3)
    xs = np.arange(5, dtype=float)
    ys = np.arange(5, dtype=float)
    hole = {(2, 2), (2, 3), (3, 2), (3, 3)}
    pts = [(x, y) for y in ys for x in xs if (int(x), int(y)) not in hole]
    xy = np.asarray(pts, dtype=float)

    # Geometry-based interior at n=1
    pos_geo = set(interior_indices_geometry(xy, 1).tolist())

    # Expected: start from ordinal interior on the full grid then remove any tile adjacent
    # (4-neighborhood) to a hole or the outer boundary.
    def idx_of(x: int, y: int) -> int:
        # Index in row-major order among kept points (skipping holes)
        # Reconstruct by scanning pts list
        for i, (xx, yy) in enumerate(pts):
            if int(xx) == x and int(yy) == y:
                return i
        raise AssertionError("point not found")

    interior_candidates = [(x, y) for y in range(1, 4) for x in range(1, 4) if (x, y) not in hole]

    def occupied(x: int, y: int) -> bool:
        return 0 <= x < 5 and 0 <= y < 5 and (x, y) not in hole

    expected_geo = set()
    for x, y in interior_candidates:
        if occupied(x - 1, y) and occupied(x + 1, y) and occupied(x, y - 1) and occupied(x, y + 1):
            expected_geo.add(idx_of(x, y))

    assert pos_geo == expected_geo


@pytest.mark.parametrize(
    "xy, n, exc",
    [
        (np.zeros((10,)), 1, ValueError),  # wrong shape
        (np.zeros((3, 3)), 1, ValueError),  # wrong second dimension
        (_grid_xy(3, 3), -1, ValueError),  # negative n
    ],
)
def test_interior_indices_error_cases(xy: np.ndarray, n: int, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        tiles_at_least_n_steps_from_edges(xy, n)
