from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from tifffile import imwrite as real_imwrite
from typer.testing import CliRunner

from fishtools.segment import extract
from fishtools.utils.io import Workspace


ORIG_DEFAULT_RNG = np.random.default_rng


class MockZarrArray:
    def __init__(self, data: np.ndarray, *, channel_names: list[str]):
        self._data = data
        self.shape = data.shape
        self.ndim = data.ndim
        self.attrs = {"key": channel_names}

    def __getitem__(self, item):
        return self._data[item]


def _write_4d_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write with explicit photometric/planarconfig to avoid tifffile warnings for multi-channel
    tifffile.imwrite(
        path,
        arr,
        metadata={"axes": "ZCYX"},
        photometric="minisblack",
        planarconfig="separate",
    )


def _read_axes_from_tif(path: Path) -> str | None:
    with tifffile.TiffFile(path) as tif:
        # Prefer shaped_metadata if present
        md = getattr(tif, "shaped_metadata", None)
        if md and isinstance(md, list) and md and isinstance(md[0], dict):
            axes = md[0].get("axes") or md[0].get("Axes")
            if isinstance(axes, str):
                return axes
        # Fallback: parse ImageDescription JSON
        desc_tag = tif.pages[0].tags.get("ImageDescription")
        if desc_tag is not None:
            try:
                meta = json.loads(desc_tag.value)
                axes = meta.get("axes") or meta.get("Axes")
                if isinstance(axes, str):
                    return axes
            except Exception:
                return None
    return None


def _make_workspace(
    tmp_path: Path, roi: str = "cortex", cb: str = "cb1"
) -> tuple[Path, Path, list[Path]]:
    root = tmp_path / "workspace"
    reg_dir = root / "analysis" / "deconv" / f"registered--{roi}+{cb}"
    files = []
    # Small synthetic 4D stack (Z,C,Y,X)
    z, c, y, x = 3, 2, 16, 32
    arr = (np.random.default_rng(0).integers(0, 2000, size=(z, c, y, x))).astype(
        np.uint16
    )
    f = reg_dir / "reg-000.tif"
    _write_4d_tif(f, arr)
    files.append(f)
    return root, reg_dir, files


def _make_workspace_with_fused_zarr(
    tmp_path: Path,
    roi: str = "cortex",
    cb: str = "cb1",
    *,
    spatial_shape: tuple[int, int] = (600, 640),
) -> tuple[Path, Path, np.ndarray, list[str]]:
    root = tmp_path / "workspace"
    fused_dir = root / "analysis" / "deconv" / f"stitch--{roi}+{cb}"
    store = fused_dir / extract.FILE_NAME
    store.mkdir(parents=True, exist_ok=True)

    z, c = 4, 3
    y, x = spatial_shape
    data = np.arange(z * y * x * c, dtype=np.uint16).reshape(z, y, x, c)
    channel_names = ["polyA", "reddot", "spots"]
    return root, store, data, channel_names


def _make_workspace_with_tiff_and_zarr(
    tmp_path: Path,
    roi: str = "cortex",
    cb: str = "cb1",
    *,
    spatial_shape: tuple[int, int] = (600, 640),
) -> tuple[Path, np.ndarray, np.ndarray, list[str]]:
    root = tmp_path / "workspace"

    reg_dir = root / "analysis" / "deconv" / f"registered--{roi}+{cb}"
    z, c = 3, 2
    y, x = spatial_shape
    tiff_data = (np.ones((z, c, y, x), dtype=np.uint16) * 1111)
    _write_4d_tif(reg_dir / "reg-000.tif", tiff_data)

    fused_dir = root / "analysis" / "deconv" / f"stitch--{roi}+{cb}"
    fused_dir.mkdir(parents=True, exist_ok=True)
    store = fused_dir / extract.FILE_NAME
    store.mkdir(parents=True, exist_ok=True)
    zarr_data = np.arange(z * y * x * c, dtype=np.uint16).reshape(z, y, x, c)
    channel_names = ["polyA", "reddot"]
    return root, tiff_data, zarr_data, channel_names


def _make_parameterized_workspace(
    tmp_path: Path, roi: str = "cortex", cb: str = "cb1", pattern: str = "z_unique"
) -> tuple[Path, Path, list[Path], np.ndarray]:
    """Create workspace with parameterized test images.

    pattern types:
    - "z_unique": Different pattern at each Z with unique values
    - "gradient": Linear gradient in Z direction
    - "checkerboard": 3D checkerboard pattern

    Returns: (root, reg_dir, files, expected_array)
    """
    root = tmp_path / "workspace"
    reg_dir = root / "analysis" / "deconv" / f"registered--{roi}+{cb}"
    files = []

    z, c, y, x = 6, 2, 16, 32  # More Z slices for better testing
    arr = np.zeros((z, c, y, x), dtype=np.uint16)

    if pattern == "z_unique":
        # Each Z slice has unique values for exact verification
        for zi in range(z):
            base_value = 1000 * (zi + 1)
            if zi % 3 == 0:  # Horizontal lines
                for yi in [4, 8, 12]:
                    arr[zi, :, yi, :] = base_value
            elif zi % 3 == 1:  # Vertical lines
                for xi in [8, 16, 24]:
                    arr[zi, :, :, xi] = base_value + 100
            else:  # Diagonal pattern
                for i in range(min(y, x)):
                    if i < y and i < x:
                        arr[zi, :, i, i] = base_value + 200
    elif pattern == "gradient":
        # Linear gradient in Z
        for zi in range(z):
            arr[zi, :, :, :] = (zi + 1) * 1000
    elif pattern == "checkerboard":
        # 3D checkerboard
        block_size = 4
        for zi in range(z):
            for yi in range(y):
                for xi in range(x):
                    if (
                        (zi // block_size + yi // block_size + xi // block_size) % 2
                    ) == 0:
                        arr[zi, :, yi, xi] = 5000
                    else:
                        arr[zi, :, yi, xi] = 1000

    f = reg_dir / "reg-000.tif"
    _write_4d_tif(f, arr)
    files.append(f)
    return root, reg_dir, files, arr


@patch("fishtools.segment.extract.imwrite")
@patch("fishtools.segment.extract.unsharp_all")
def test_cmd_extract_z_basic(mock_unsharp, mock_imwrite, tmp_path: Path) -> None:
    # Mock unsharp_mask to return input unchanged
    mock_unsharp.side_effect = lambda img, **kwargs: img

    # Capture what's being written to files
    written_data = {}

    def capture_imwrite(file_path, data, **kwargs):
        written_data[Path(file_path).name] = data.copy()
        # Still write the actual file for other assertions
        return real_imwrite(file_path, data, **kwargs)

    mock_imwrite.side_effect = capture_imwrite

    root, _reg_dir, _files, expected = _make_parameterized_workspace(
        tmp_path, roi="roiA", cb="cb1", pattern="z_unique"
    )

    extract.cmd_extract(
        "z",
        root,
        roi="roiA",
        codebook="cb1",
        channels="0,1",
        threads=1,
        dz=1,
    )

    ws = Workspace(root)
    out_dir = ws.segment("roiA", "cb1") / "extract_z"
    outs = sorted(out_dir.glob("*_z*.tif"))
    assert len(outs) == 6  # one per Z (6 slices)

    # Verify filenames are correct
    expected_filenames = [f"reg-000_z{zi:04d}.tif" for zi in range(6)]
    actual_filenames = [f.name for f in outs]
    assert actual_filenames == expected_filenames, (
        f"Filenames don't match: {actual_filenames}"
    )

    # Verify data captured by mock
    assert len(written_data) == 6, f"Expected 6 files written, got {len(written_data)}"

    # Check each captured slice
    for zi in range(6):
        filename = f"reg-000_z{zi:04d}.tif"
        assert filename in written_data, f"File {filename} not in written data"

        captured = written_data[filename]
        expected_slice = expected[zi, :, :, :]

        # Debug output for first slice if mismatch
        if zi == 0 and not np.array_equal(captured, expected_slice):
            print(f"\nZ={zi} mismatch:")
            print(
                f"Captured shape: {captured.shape}, Expected shape: {expected_slice.shape}"
            )
            print(f"Captured unique values: {np.unique(captured)}")
            print(f"Expected unique values: {np.unique(expected_slice)}")
            print(
                f"Max diff: {np.max(np.abs(captured.astype(float) - expected_slice.astype(float)))}"
            )

        np.testing.assert_array_equal(
            captured,
            expected_slice,
            err_msg=f"Z-slice {zi} from mock doesn't match expected",
        )

    # Also verify files were actually written and readable
    for zi, out_file in enumerate(outs):
        arr = tifffile.imread(out_file)
        assert arr.ndim == 3 and arr.shape[0] == 2  # (C,Y,X)
        assert arr.dtype == np.uint16


def test_cli_extract_z_smoke(tmp_path: Path) -> None:
    root, _reg_dir, _files, _expected = _make_parameterized_workspace(
        tmp_path, roi="roi_cli_z", cb="cb1", pattern="gradient"
    )

    runner = CliRunner()
    result = runner.invoke(
        extract.app,
        [
            "z",
            str(root),
            "--roi",
            "roi_cli_z",
            "--codebook",
            "cb1",
            "--channels",
            "0,1",
            "--threads",
            "1",
            "--dz",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output

    ws = Workspace(root)
    out_dir = ws.segment("roi_cli_z", "cb1") / "extract_z"
    assert out_dir.exists()
    assert any(out_dir.glob("*.tif"))


@patch("fishtools.segment.extract.imwrite")
@patch("fishtools.segment.extract.unsharp_all")
def test_cmd_extract_z_from_zarr(
    mock_unsharp, mock_imwrite, tmp_path: Path
) -> None:
    mock_unsharp.side_effect = lambda img, **kwargs: img

    roi = "roiZarr"
    cb = "cb1"
    root, store, data, channel_names = _make_workspace_with_fused_zarr(
        tmp_path,
        roi=roi,
        cb=cb,
        spatial_shape=(64, 72),
    )

    captured: dict[str, np.ndarray] = {}

    def capture_imwrite(file_path, arr, **kwargs):
        captured[Path(file_path).name] = arr.copy()
        return real_imwrite(file_path, arr, **kwargs)

    mock_imwrite.side_effect = capture_imwrite

    tile_size = 32
    default_rng = ORIG_DEFAULT_RNG

    def fake_rng(*args, **kwargs):
        return default_rng(0)

    mock_zarr = MockZarrArray(data, channel_names=channel_names)

    with patch.object(extract, "ZARR_TILE_SIZE", tile_size), patch(
        "fishtools.segment.extract._open_zarr_array", side_effect=lambda *_: mock_zarr
    ), patch("numpy.random.default_rng", side_effect=fake_rng):
        extract.cmd_extract(
            "z",
            root,
            roi=roi,
            codebook=cb,
            channels="auto",
            threads=1,
            n=2,
        )

    ws = Workspace(root)
    out_dir = ws.segment(roi, cb) / "extract_z"
    outs = sorted(out_dir.glob("*.tif"))

    tile_origins = extract._compute_tile_origins(
        data.shape,
        tile_size=tile_size,
        n_tiles=2,
        crop=0,
    )
    z_candidates = list(range(0, data.shape[0], 1))
    expected_rng = default_rng(0)
    sampled = [int(expected_rng.choice(z_candidates)) for _ in tile_origins]
    coord_width = len(str(max(data.shape[1], data.shape[2])))

    expected_names = [
        f"fused_y{y0:0{coord_width}d}_x{x0:0{coord_width}d}_z{zi:02d}.tif"
        for (y0, x0), zi in zip(tile_origins, sampled)
    ]
    assert [p.name for p in outs] == expected_names

    first_name = expected_names[0]
    first = captured[first_name]
    (y0, x0), z0 = tile_origins[0], sampled[0]
    expected = np.moveaxis(
        data[z0, y0 : y0 + tile_size, x0 : x0 + tile_size, :2],
        -1,
        0,
    )
    expected = np.clip(expected, 0, 65530).astype(np.uint16)
    np.testing.assert_array_equal(first, expected)


@patch("fishtools.segment.extract.imwrite")
@patch("fishtools.segment.extract.unsharp_all")
def test_cmd_extract_ortho_from_zarr_random_positions(
    mock_unsharp, mock_imwrite, tmp_path: Path
) -> None:
    mock_unsharp.side_effect = lambda img, **kwargs: img

    roi = "roiZarrOrtho"
    cb = "cb1"
    root, store, data, channel_names = _make_workspace_with_fused_zarr(
        tmp_path,
        roi=roi,
        cb=cb,
        spatial_shape=(64, 80),
    )

    captured: dict[str, np.ndarray] = {}

    def capture_imwrite(file_path, arr, **kwargs):
        captured[Path(file_path).name] = arr.copy()
        return real_imwrite(file_path, arr, **kwargs)

    mock_imwrite.side_effect = capture_imwrite

    default_rng = ORIG_DEFAULT_RNG

    def fake_rng(*args, **kwargs):
        return default_rng(0)

    mock_zarr = MockZarrArray(data, channel_names=channel_names)

    with patch(
        "fishtools.segment.extract._open_zarr_array", side_effect=lambda *_: mock_zarr
    ), patch("numpy.random.default_rng", side_effect=fake_rng):
        extract.cmd_extract(
            "ortho",
            root,
            roi=roi,
            codebook=cb,
            channels="auto",
            threads=1,
            n=3,
            anisotropy=2,
        )

    ws = Workspace(root)
    out_dir = ws.segment(roi, cb) / "extract_ortho"
    outs = sorted(out_dir.glob("*.tif"))
    assert outs, "No ortho outputs produced"

    expected_rng = default_rng(0)
    expected_y = extract._sample_positions(data.shape[1], crop=0, count=3, rng=expected_rng)
    expected_x = extract._sample_positions(data.shape[2], crop=0, count=3, rng=expected_rng)

    zx_names = [f"fused_orthozx-{y}.tif" for y in expected_y]
    zy_names = [f"fused_orthozy-{x}.tif" for x in expected_x]
    assert sorted(p.name for p in outs if "orthozx" in p.name) == sorted(zx_names)
    assert sorted(p.name for p in outs if "orthozy" in p.name) == sorted(zy_names)

    sample_name = zx_names[0]
    sample_arr = captured[sample_name]
    assert sample_arr.shape[0] == 2  # channels
    assert sample_arr.shape[1] == data.shape[0] * 2  # anisotropy scaling
    assert sample_arr.shape[2] == data.shape[2]
    assert sample_arr.dtype == np.uint16


@patch("fishtools.segment.extract.progress_bar")
@patch("fishtools.segment.extract.unsharp_all")
def test_cmd_extract_z_from_zarr_tracks_progress(
    mock_unsharp, mock_progress_bar, tmp_path: Path
) -> None:
    mock_unsharp.side_effect = lambda img, **kwargs: img

    calls: dict[str, int | None] = {"total": None, "count": 0}

    @contextmanager
    def fake_progress(n: int):
        calls["total"] = n

        def _increment() -> None:
            calls["count"] = int(calls["count"]) + 1

        yield _increment

    mock_progress_bar.side_effect = fake_progress

    roi = "roiZarrProgress"
    cb = "cb1"
    root, store, data, channel_names = _make_workspace_with_fused_zarr(
        tmp_path,
        roi=roi,
        cb=cb,
        spatial_shape=(64, 72),
    )

    tile_size = 32
    mock_zarr = MockZarrArray(data, channel_names=channel_names)
    default_rng = ORIG_DEFAULT_RNG

    def fake_rng(*args, **kwargs):
        return default_rng(0)

    with patch.object(extract, "ZARR_TILE_SIZE", tile_size), patch(
        "fishtools.segment.extract._open_zarr_array", side_effect=lambda *_: mock_zarr
    ), patch("numpy.random.default_rng", side_effect=fake_rng):
        extract.cmd_extract(
            "z",
            root,
            roi=roi,
            codebook=cb,
            channels="auto",
            threads=1,
            n=3,
        )

        tile_origins = extract._compute_tile_origins(
            data.shape,
            tile_size=tile_size,
            n_tiles=3,
            crop=0,
        )
        expected_outputs = len(tile_origins)

    assert calls["total"] == expected_outputs
    assert calls["count"] == expected_outputs


@patch("fishtools.segment.extract.progress_bar")
@patch("fishtools.segment.extract.unsharp_all")
def test_cmd_extract_ortho_from_zarr_tracks_progress(
    mock_unsharp, mock_progress_bar, tmp_path: Path
) -> None:
    mock_unsharp.side_effect = lambda img, **kwargs: img

    calls: dict[str, int | None] = {"total": None, "count": 0}

    @contextmanager
    def fake_progress(n: int):
        calls["total"] = n

        def _increment() -> None:
            calls["count"] = int(calls["count"]) + 1

        yield _increment

    mock_progress_bar.side_effect = fake_progress

    roi = "roiZarrOrthoProgress"
    cb = "cb1"
    root, store, data, channel_names = _make_workspace_with_fused_zarr(
        tmp_path,
        roi=roi,
        cb=cb,
        spatial_shape=(64, 80),
    )

    default_rng = ORIG_DEFAULT_RNG

    def fake_rng(*args, **kwargs):
        return default_rng(0)

    mock_zarr = MockZarrArray(data, channel_names=channel_names)

    with patch(
        "fishtools.segment.extract._open_zarr_array", side_effect=lambda *_: mock_zarr
    ), patch("numpy.random.default_rng", side_effect=fake_rng):
        extract.cmd_extract(
            "ortho",
            root,
            roi=roi,
            codebook=cb,
            channels="auto",
            threads=1,
            n=4,
            anisotropy=2,
        )

        expected_rng = default_rng(0)
        expected_y = extract._sample_positions(data.shape[1], crop=0, count=4, rng=expected_rng)
        expected_x = extract._sample_positions(data.shape[2], crop=0, count=4, rng=expected_rng)
        expected_outputs = len(expected_y) + len(expected_x)

    assert calls["total"] == expected_outputs
    assert calls["count"] == expected_outputs


@patch("fishtools.segment.extract.imwrite")
@patch("fishtools.segment.extract.unsharp_all")
def test_cmd_extract_z_with_zarr_flag_prefers_zarr(
    mock_unsharp, mock_imwrite, tmp_path: Path
) -> None:
    mock_unsharp.side_effect = lambda img, **kwargs: img

    roi = "roiCombo"
    cb = "cb1"
    root, _tiff_data, zarr_data, channel_names = _make_workspace_with_tiff_and_zarr(
        tmp_path,
        roi=roi,
        cb=cb,
        spatial_shape=(64, 72),
    )

    captured: dict[str, np.ndarray] = {}

    def capture_imwrite(file_path, arr, **kwargs):
        captured[Path(file_path).name] = arr.copy()
        return real_imwrite(file_path, arr, **kwargs)

    mock_imwrite.side_effect = capture_imwrite

    tile_size = 32
    default_rng = ORIG_DEFAULT_RNG

    def fake_rng(*args, **kwargs):
        return default_rng(0)

    mock_zarr = MockZarrArray(zarr_data, channel_names=channel_names)

    with patch.object(extract, "ZARR_TILE_SIZE", tile_size), patch(
        "fishtools.segment.extract._open_zarr_array", side_effect=lambda *_: mock_zarr
    ), patch("numpy.random.default_rng", side_effect=fake_rng):
        extract.cmd_extract(
            "z",
            root,
            roi=roi,
            codebook=cb,
            channels="0,1",
            threads=1,
            use_zarr=True,
            n=2,
        )

    ws = Workspace(root)
    out_dir = ws.segment(roi, cb) / "extract_z"
    outs = sorted(out_dir.glob("*.tif"))
    assert outs, "No output files produced"

    tile_origins = extract._compute_tile_origins(
        zarr_data.shape,
        tile_size=tile_size,
        n_tiles=2,
        crop=0,
    )
    coord_width = len(str(max(zarr_data.shape[1], zarr_data.shape[2])))
    z_candidates = list(range(0, zarr_data.shape[0]))
    expected_rng = default_rng(0)
    sampled = [int(expected_rng.choice(z_candidates)) for _ in tile_origins]
    expected_names = [
        f"fused_y{y0:0{coord_width}d}_x{x0:0{coord_width}d}_z{zi:02d}.tif"
        for (y0, x0), zi in zip(tile_origins, sampled)
    ]
    assert [p.name for p in outs] == expected_names

    first_name = expected_names[0]
    first = captured[first_name]
    (y0, x0), z0 = tile_origins[0], sampled[0]
    expected = np.moveaxis(
        zarr_data[z0, y0 : y0 + tile_size, x0 : x0 + tile_size, :2],
        -1,
        0,
    )
    expected = np.clip(expected, 0, 65530).astype(np.uint16)
    np.testing.assert_array_equal(first, expected)


@patch("fishtools.segment.extract.unsharp_all")
def test_cmd_extract_z_with_zarr_flag_requires_store(mock_unsharp, tmp_path: Path) -> None:
    mock_unsharp.side_effect = lambda img, **kwargs: img

    roi = "roiMissing"
    cb = "cb1"
    root, _reg_dir, _files = _make_workspace(tmp_path, roi=roi, cb=cb)

    with pytest.raises(FileNotFoundError, match="Requested Zarr input"):
        extract.cmd_extract(
            "z",
            root,
            roi=roi,
            codebook=cb,
            channels="0,1",
            threads=1,
            use_zarr=True,
        )


@patch("fishtools.segment.extract.unsharp_all")
def test_cmd_extract_ortho_basic(mock_unsharp, tmp_path: Path) -> None:
    # Mock unsharp_mask to return input unchanged
    mock_unsharp.side_effect = lambda img, **kwargs: img

    root, _reg_dir, _files, expected = _make_parameterized_workspace(
        tmp_path, roi="roiB", cb="cb1", pattern="gradient"
    )
    out = tmp_path / "ortho_out"

    extract.cmd_extract(
        "ortho",
        root,
        roi="roiB",
        codebook="cb1",
        out=out,
        channels="0,1",
        threads=1,
        n=2,
        anisotropy=3,
    )

    outs = sorted(out.glob("*.tif"))
    # Expect 2 positions × (zx, zy) = 4 files
    assert len(outs) == 4

    # Calculate expected Y positions (10% and 90% of Y dimension)
    y_dim = expected.shape[2]  # 16
    expected_positions = np.linspace(int(0.1 * y_dim), int(0.9 * y_dim), 2).astype(int)

    for f in outs:
        arr = tifffile.imread(f)
        assert arr.ndim == 3 and arr.shape[0] == 2  # channels preserved
        assert arr.dtype == np.uint16
        axes = _read_axes_from_tif(f)
        assert axes in {"CZX", "CZY"}

        # Extract position from filename
        pos = int(f.stem.split("-")[-1])
        assert pos in expected_positions

        # Check anisotropy scaling of Z dimension (input Z=6, anisotropy=3)
        if "orthozx" in f.name:
            # CZX: Z-dim is axis=1, X is axis=2
            assert arr.shape[1] == 6 * 3  # Z * anisotropy
            assert arr.shape[2] == 32  # X unchanged

            # Verify gradient pattern preserved (should see increasing values in Z)
            for zi in range(6):
                # Account for zoom interpolation
                zoomed_idx = zi * 3  # Approximate center of zoomed region
                if zoomed_idx < arr.shape[1]:
                    # Gradient value should be approximately (zi+1)*1000
                    expected_val = (zi + 1) * 1000
                    actual_val = np.mean(arr[:, zoomed_idx : zoomed_idx + 3, :])
                    assert abs(actual_val - expected_val) < 500, (
                        f"Gradient not preserved at Z={zi}"
                    )

        elif "orthozy" in f.name:
            # CZY: Z-dim is axis=1, Y is axis=2
            assert arr.shape[1] == 6 * 3  # Z * anisotropy
            assert arr.shape[2] == 16  # Y unchanged


def test_extract_z_stride_and_positions(tmp_path: Path) -> None:
    # Build a stack with bright impulses per Z so unsharp filtering preserves non-zeros
    root = tmp_path / "workspace"
    roi = "roiZ"
    cb = "cb1"
    reg = root / "analysis" / "deconv" / f"registered--{roi}+{cb}"
    reg.mkdir(parents=True, exist_ok=True)

    z, c, y, x = 4, 2, 8, 8
    arr = np.zeros((z, c, y, x), dtype=np.uint16)
    for zi in range(z):
        arr[zi, 0, 3, 5] = 60000  # ch0 impulse
        arr[zi, 1, 4, 3] = 45000  # ch1 impulse at different coord
    f = reg / "reg-000.tif"
    _write_4d_tif(f, arr)

    extract.cmd_extract(
        "z",
        root,
        roi=roi,
        codebook=cb,
        channels="0,1",
        dz=2,
        threads=1,
    )

    ws = Workspace(root)
    out_dir = ws.segment(roi, cb) / "extract_z"
    outs = sorted(out_dir.glob("reg-000_z*.tif"))
    # Expect slices at z=0 and z=2
    assert [p.name for p in outs] == ["reg-000_z0000.tif", "reg-000_z0002.tif"]
    a0 = tifffile.imread(outs[0])
    a2 = tifffile.imread(outs[1])
    # Shapes and impulse positions preserved
    assert a0.shape == (2, y, x) and a2.shape == (2, y, x)
    assert a0[0, 3, 5] > 0 and a0[1, 4, 3] > 0
    assert a2[0, 3, 5] > 0 and a2[1, 4, 3] > 0


def test_extract_ortho_positions_values_anis1(tmp_path: Path) -> None:
    # Build a stack with bright lines at target positions; anisotropy=1 for value locality
    root = tmp_path / "workspace"
    roi = "roiO"
    cb = "cb1"
    reg = root / "analysis" / "deconv" / f"registered--{roi}+{cb}"
    reg.mkdir(parents=True, exist_ok=True)

    z, c, y, x = 3, 2, 10, 12
    base = np.zeros((z, c, y, x), dtype=np.uint16)
    # Compute positions first, then write lines at those rows/columns
    y0 = int(0.1 * y)
    y1 = int(0.9 * y)
    positions = np.linspace(y0, y1, 2).astype(int).tolist()
    for zi in range(z):
        # Bright horizontal lines (affects ZX at y=pos)
        for pos in positions:
            base[zi, :, pos, :] = 50000
            # Bright vertical lines (affects ZY at x=pos)
            base[zi, :, :, pos] = 45000
    _write_4d_tif(reg / "reg-000.tif", base)

    extract.cmd_extract(
        "ortho",
        root,
        roi=roi,
        codebook=cb,
        channels="0,1",
        n=2,
        anisotropy=1,
        threads=1,
    )

    # Expected positions are evenly spaced between 10% and 90% of Y
    # Default output location under segment dir
    ws = Workspace(root)
    out_dir = ws.segment(roi, cb) / "extract_ortho"
    outs = {p.name: tifffile.imread(p) for p in out_dir.glob("*.tif")}

    # Check both positions and both dims
    for pos in positions:
        zx_name = f"reg-000_orthozx-{pos}.tif"
        zy_name = f"reg-000_orthozy-{pos}.tif"
        assert zx_name in outs and zy_name in outs
        a_zx = outs[zx_name]  # (C,Z,X)
        a_zy = outs[zy_name]  # (C,Z,Y)
        # For each z, confirm non-zero along the extracted row/column for both channels
        for zi in range(z):
            assert (a_zx[0, zi, :] > 0).all() and (a_zx[1, zi, :] > 0).all()
            assert (a_zy[0, zi, :] > 0).all() and (a_zy[1, zi, :] > 0).all()


def test_cli_extract_ortho_smoke(tmp_path: Path) -> None:
    root, _reg_dir, _files, _expected = _make_parameterized_workspace(
        tmp_path, roi="roi_cli_ortho", cb="cb1", pattern="gradient"
    )

    runner = CliRunner()
    result = runner.invoke(
        extract.app,
        [
            "ortho",
            str(root),
            "--roi",
            "roi_cli_ortho",
            "--codebook",
            "cb1",
            "--channels",
            "0,1",
            "--threads",
            "1",
            "--n",
            "2",
            "--anisotropy",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    ws = Workspace(root)
    out_dir = ws.segment("roi_cli_ortho", "cb1") / "extract_ortho"
    assert out_dir.exists()
    assert any(out_dir.glob("*.tif"))


def test_resolve_channels_with_names(tmp_path: Path) -> None:
    # Create a stack with channel names in shaped metadata
    root, reg_dir, files = _make_workspace(tmp_path, roi="roiC", cb="cb1")
    f = files[0]
    arr = tifffile.imread(f)
    # Overwrite with names in metadata
    tifffile.imwrite(f, arr, metadata={"axes": "ZCYX", "key": ["polyA", "reddot"]})

    # Name-based channels
    extract.cmd_extract(
        "z",
        root,
        roi="roiC",
        codebook="cb1",
        channels="polyA,reddot",
        threads=1,
        dz=1,
    )
    ws = Workspace(root)
    out_dir = ws.segment("roiC", "cb1") / "extract_z"
    out = sorted(out_dir.glob("*_z0000.tif"))[0]
    md_axes = _read_axes_from_tif(out)
    assert md_axes == "CYX"
    with tifffile.TiffFile(out) as tif:
        meta = tif.pages[0].tags.get("ImageDescription").value
        j = json.loads(meta)
        assert j.get("channel_names") == ["polyA", "reddot"]
        assert j.get("channels_arg") == "polyA,reddot"


@pytest.mark.parametrize("dz", [1, 2, 3])
@patch("fishtools.segment.extract.unsharp_all")
def test_extract_z_dz_parameter(mock_unsharp, tmp_path: Path, dz: int) -> None:
    """Test that dz parameter correctly controls Z-slice extraction interval."""
    mock_unsharp.side_effect = lambda img, **kwargs: img

    root, _reg_dir, _files, expected = _make_parameterized_workspace(
        tmp_path, roi="roiDZ", cb="cb1", pattern="z_unique"
    )

    extract.cmd_extract(
        "z",
        root,
        roi="roiDZ",
        codebook="cb1",
        channels="0,1",
        dz=dz,
        threads=1,
    )

    ws = Workspace(root)
    out_dir = ws.segment("roiDZ", "cb1") / "extract_z"
    outs = sorted(out_dir.glob("*_z*.tif"))

    # Should extract every dz'th slice starting from 0
    expected_count = len(range(0, 6, dz))  # 6 total Z slices
    assert len(outs) == expected_count, (
        f"Expected {expected_count} files for dz={dz}, got {len(outs)}"
    )

    # Verify correct Z indices in filenames
    expected_indices = list(range(0, 6, dz))
    actual_indices = [int(f.stem.split("_z")[1]) for f in outs]
    assert actual_indices == expected_indices, f"Wrong Z indices for dz={dz}"

    # Verify each slice has correct content pattern
    for i, (zi, out_file) in enumerate(zip(expected_indices, outs)):
        arr = tifffile.imread(out_file)
        expected_slice = expected[zi, :, :, :]

        # Check the pattern is preserved (allowing for small edge effects from unsharp)
        # The z_unique pattern has specific non-zero locations
        base_value = 1000 * (zi + 1)

        if zi % 3 == 0:  # Horizontal lines pattern
            # Check horizontal lines at y=[4,8,12]
            for yi in [4, 8, 12]:
                line_values = arr[:, yi, :]
                # Most values on the line should be near base_value
                assert np.median(line_values) > base_value - 100, (
                    f"dz={dz}, Z={zi}: Horizontal line at y={yi} not preserved"
                )
        elif zi % 3 == 1:  # Vertical lines pattern
            # Check vertical lines at x=[8,16,24]
            for xi in [8, 16, 24]:
                line_values = arr[:, :, xi]
                # Most values on the line should be near base_value+100
                assert np.median(line_values) > base_value, (
                    f"dz={dz}, Z={zi}: Vertical line at x={xi} not preserved"
                )
        else:  # Diagonal pattern
            # Check diagonal values
            diag_values = [arr[:, i, i] for i in range(min(16, 32)) if i < 16]
            diag_median = np.median([v for vals in diag_values for v in vals])
            assert diag_median > base_value + 100, (
                f"dz={dz}, Z={zi}: Diagonal pattern not preserved"
            )


@patch("fishtools.segment.extract.imwrite")
@patch("fishtools.segment.extract.unsharp_all")
def test_wrong_slice_extraction_should_fail(
    mock_unsharp, mock_imwrite, tmp_path: Path
) -> None:
    """Intentionally failing test to verify our slice verification works.

    This test deliberately writes the wrong Z-slice to verify that our
    test infrastructure correctly detects when the wrong slice is extracted.
    """
    mock_unsharp.side_effect = lambda img, **kwargs: img

    # Capture writes and deliberately swap slices
    written_data = {}

    def capture_and_corrupt_imwrite(file_path, data, **kwargs):
        filename = Path(file_path).name
        # Deliberately write wrong data - if it's z0000, write z0001 data instead
        if "z0000" in filename:
            # Create corrupted data (all ones instead of expected pattern)
            corrupted = np.ones_like(data) * 9999
            written_data[filename] = corrupted
            return real_imwrite(file_path, corrupted, **kwargs)
        else:
            written_data[filename] = data.copy()
            return real_imwrite(file_path, data, **kwargs)

    mock_imwrite.side_effect = capture_and_corrupt_imwrite

    root, _reg_dir, _files, expected = _make_parameterized_workspace(
        tmp_path, roi="roiFAIL", cb="cb1", pattern="z_unique"
    )

    extract.cmd_extract(
        "z",
        root,
        roi="roiFAIL",
        codebook="cb1",
        channels="0,1",
        threads=1,
        dz=1,
    )

    # Now verify that our test would catch the corruption
    ws = Workspace(root)
    out_dir = ws.segment("roiFAIL", "cb1") / "extract_z"
    outs = sorted(out_dir.glob("*_z*.tif"))

    # This should fail because z0000 has wrong data
    z0_file = outs[0]
    assert "z0000" in z0_file.name
    arr = tifffile.imread(z0_file)

    # This assertion should FAIL because we corrupted the data
    # Expected: horizontal lines at y=[4,8,12] with value 1000
    # Actual: all 9999s
    with pytest.raises(AssertionError, match="does not match expected pattern"):
        # Check if horizontal lines exist at expected positions
        for yi in [4, 8, 12]:
            line_values = arr[:, yi, :]
            expected_value = 1000  # Z=0 should have value 1000
            actual_median = np.median(line_values)
            # This should fail because actual_median will be 9999
            assert abs(actual_median - expected_value) < 100, (
                f"Z=0: Line at y={yi} does not match expected pattern. "
                f"Expected ~{expected_value}, got {actual_median:.0f}"
            )


def test_max_from_cli_appends_and_sets_metadata(tmp_path: Path) -> None:
    # Workspace with two codebooks
    root = tmp_path / "workspace"
    roi = "roiD"
    cb_main = "cb1"
    cb_aux = "cb2"
    reg_main = root / "analysis" / "deconv" / f"registered--{roi}+{cb_main}"
    reg_aux = root / "analysis" / "deconv" / f"registered--{roi}+{cb_aux}"
    reg_main.mkdir(parents=True, exist_ok=True)
    reg_aux.mkdir(parents=True, exist_ok=True)

    z, c, y, x = 2, 2, 10, 12
    file = reg_main / "reg-000.tif"
    file_aux = reg_aux / file.name
    _write_4d_tif(file, (np.arange(z * c * y * x, dtype=np.uint16).reshape(z, c, y, x)))
    _write_4d_tif(
        file_aux, (np.arange(z * c * y * x, dtype=np.uint16).reshape(z, c, y, x))
    )

    extract.cmd_extract(
        "z",
        root,
        roi=roi,
        codebook=cb_main,
        channels="0,1",
        max_from=cb_aux,
        threads=1,
        dz=1,
    )
    ws = Workspace(root)
    out_dir = ws.segment(roi, cb_main) / "extract_z"
    outs = sorted(out_dir.glob("*_z*.tif"))
    assert outs, "no outputs written"
    arr = tifffile.imread(outs[0])
    # Expect original selected channels (2) + 1 appended max channel = 3
    assert arr.shape[0] == 3
    meta = json.loads(
        tifffile.TiffFile(outs[0]).pages[0].tags["ImageDescription"].value
    )
    assert meta.get("channel_names") and meta["channel_names"][-1] == "max_from"
