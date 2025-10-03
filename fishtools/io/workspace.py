import io
import json
import os
import re
import shutil
import warnings
from contextlib import contextmanager, redirect_stderr, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, overload

import numpy as np
import numpy.typing as npt
import requests
from loguru import logger
from tifffile import TiffFile, imread
from tifffile import imwrite as tifffile_imwrite
from tqdm.auto import tqdm

from fishtools.preprocess.tileconfig import TileConfiguration


class CorruptedTiffError(RuntimeError):
    """Raised when a TIFF file cannot be read due to corruption."""

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"File {path} is corrupted. Please check the file.")
        self.path = path
        self.__cause__ = cause


def download(url: str, path: str | Path, name: str | None = None) -> None:
    with requests.get(url, stream=True, allow_redirects=True, timeout=30) as r:
        if r.status_code >= 400:
            r.raise_for_status()
            raise RuntimeError(f"Request to {url} returned status code {r.status_code}")

        size = int(r.headers.get("Content-Length", 0))
        if name is None:
            name = url.split("/")[-1]

        path = Path(path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        path = path / name
        if path.exists():
            if size == path.stat().st_size:
                logger.info(f"File {path} already exists. Skipping.")
                return
            else:
                logger.warning(f"File {path} already exists but size is different. Redownloading.")
        else:
            logger.info(f"Downloading {url}")

        with tqdm.wrapattr(r.raw, "read", total=size) as r_raw:
            with open(f"{path}.tmp", "wb") as f:
                shutil.copyfileobj(r_raw, f)
                os.rename(f"{path}.tmp", path)

    logger.info(f"Finished downloading {url}")


def get_file_name(url: str) -> str:
    return url.split("/")[-1]


def safe_imwrite(
    path: Path | str,
    data: npt.ArrayLike,
    *,
    imwrite_func: Callable[..., Any] = tifffile_imwrite,
    partial_suffix: str = ".partial",
    mkdir: bool = True,
    **kwargs: Any,
) -> None:
    """Write TIFF data atomically by using a temporary `.partial` file.

    The data is first written to `<filename>.tif.partial` and only renamed to the
    final `.tif` path after the write succeeds, ensuring readers never observe a
    partially written TIFF. On failure the temporary file is removed.

    Args:
        path: Destination path for the final TIFF file.
        data: Array-like payload to pass to the underlying ``imwrite`` function.
        imwrite_func: Callable used to persist the data (defaults to ``tifffile.imwrite``).
        partial_suffix: Suffix appended to the filename while writing.
        mkdir: When True, create the parent directory if missing.
        **kwargs: Additional keyword arguments forwarded to ``imwrite_func``.

    Raises:
        Exception: Propagates any exception raised by the underlying writer or rename.
    """

    final_path = Path(path)
    if mkdir:
        final_path.parent.mkdir(parents=True, exist_ok=True)

    partial_path = final_path.with_name(f"{final_path.name}{partial_suffix}")

    if partial_path.exists():
        partial_path.unlink()

    try:
        imwrite_func(partial_path, data, **kwargs)
    except Exception:
        with suppress(FileNotFoundError):
            partial_path.unlink()
        raise

    try:
        partial_path.replace(final_path)
    except Exception:
        with suppress(FileNotFoundError):
            partial_path.unlink()
        raise


@contextmanager
def set_cwd(path: Path):
    cwd = Path.cwd()
    try:
        Path(path).mkdir(exist_ok=True, parents=True)
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


@dataclass
class OptimizePath:
    path: Path

    @property
    def mse(self):
        return self.path / "mse.txt"

    @property
    def scaling_factor(self):
        return self.path / "global_scale.txt"


@dataclass
class Codebook:
    path: Path

    def __post_init__(self):
        self.path = Path(self.path)
        self.codebook = json.loads(self.path.read_text())

    @property
    def name(self):
        return self.path.stem

    def to_dataframe(self, bits_as_list: bool = False):
        import polars as pl

        df = (pl.DataFrame(self.codebook).transpose(include_header=True)).rename({"column": "target"})

        if not bits_as_list:
            return df.rename({"column_0": "bit0", "column_1": "bit1", "column_2": "bit2"})

        return df.with_columns(
            concat_list=pl.concat_list("column_0", "column_1", "column_2")
            .cast(pl.List(pl.UInt8))
            .alias("bits")
        )


@dataclass
class Workspace:
    """FISH experiment workspace manager with verified directory structure.

    Provides standardized access to FISH data following the processing pipeline:
    Raw Data → Deconv → Register → Stitch → Analysis

    Directory Structure:
        workspace/
        ├── {round}--{roi}/                    # Raw imaging data
        │   ├── {round}-0001.tif
        │   └── {round}-0002.tif
        └── analysis/
            ├── deconv/
            │   ├── {round}--{roi}/            # Deconvolved images
            │   ├── registered--{roi}+{codebook}/  # Registration results
            │   ├── stitch--{roi}/             # Stitching results
            │   ├── segment--{roi}+{codebook}/ # Segmentation results
            │   ├── shifts--{roi}+{codebook}/  # Registration shifts
            │   │   └── shifts-0001.json
            │   ├── fids--{roi}/               # Fiducial markers
            │   │   └── fids-0001.tif
            │   └── opt_{codebook}/            # Optimization results
            │       ├── mse.txt
            │       └── global_scale.txt
            └── output/                        # Final analysis output

    CLI Pipeline & I/O:
        deconv:     {round}--{roi}/*.tif → analysis/deconv/{round}--{roi}/*.tif
        register:   analysis/deconv/{round}--{roi}/ → analysis/deconv/registered--{roi}+{codebook}/
        stitch:     analysis/deconv/registered--{roi}+{codebook}/ → analysis/deconv/stitch--{roi}/
        spotlook:   analysis/deconv/registered--{roi}+{codebook}/ → analysis/output/

    Key Methods:
        ws.rounds, ws.rois          # Discover available data
        ws.img(round, roi, idx)     # Access deconvolved images
        ws.regimg(roi, cb, idx)     # Access registered results
        ws.registered(roi, cb)      # Registration directory
        ws.stitch(roi)              # Stitching directory
        ws.opt(codebook)            # Optimization results

    Args:
        path: Workspace root path (auto-detects from subdirectories)
    """

    # Regex patterns for robust directory name parsing
    ROUND_ROI_PATTERN = re.compile(
        r"^([^-]+)--([^+]+)(?:\+.*)?$"
    )  # {round}--{roi} or {round}--{roi}+{suffix}
    ROI_CODEBOOK_PATTERN = re.compile(r"^[^-]+--([^+]+)(?:\+(.+))?$")  # Extract ROI and optional codebook
    NUMERIC_SORT_PATTERN = re.compile(r"^(\d+)_")
    _FORBIDDEN_ROUND_PREFIXES = (
        "10x",
        "analysis",
        "shifts",
        "stitch",
        "fid",
        "registered",
        "old",
        "basic",
    )

    path: Path

    def __str__(self) -> str:
        """Return string representation of workspace path."""
        return str(self.path)

    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        return f"Workspace({self.path})"

    def __init__(self, path: Path | str) -> None:
        """Initialize workspace with automatic path resolution.

        Automatically detects and normalizes workspace root path. If path points
        to analysis/deconv subdirectory, automatically resolves to workspace root.

        Args:
            path: Path to workspace root or any subdirectory within workspace
            deconved: Unused parameter (backward compatibility)
        """
        path = Path(path).expanduser().resolve()
        if "analysis/deconv" in path.as_posix():
            self.path = path.parent.parent
        else:
            self.path = path

    @property
    def rounds(self) -> list[str]:
        """Discover and return all available imaging rounds in the workspace."""
        return self.discover_rounds(self.path)

    @staticmethod
    def _round_sort_key(round_name: str) -> tuple[int, str]:
        head = round_name.split("_")[0]
        if head.isdigit():
            return 0, f"{int(head):08d}_{round_name}"
        return 1, round_name

    @classmethod
    def discover_rounds(cls, workspace_path: Path | str) -> list[str]:
        """Return sorted list of imaging rounds discovered under a workspace root."""

        base = Path(workspace_path).expanduser().resolve()
        search_roots = [base]
        deconv_root = base / "analysis" / "deconv"
        if deconv_root.exists():
            search_roots.append(deconv_root)

        rounds_set: set[str] = set()
        for root in search_roots:
            if not root.exists():
                continue
            for entry in root.iterdir():
                if not entry.is_dir():
                    continue
                match = cls.ROUND_ROI_PATTERN.match(entry.name)
                if not match:
                    continue
                round_name = match.group(1)
                if any(round_name.startswith(prefix) for prefix in cls._FORBIDDEN_ROUND_PREFIXES):
                    continue
                rounds_set.add(round_name)

        if not rounds_set:
            raise ValueError(f"No round subdirectories found in {base}.")

        return sorted(rounds_set, key=cls._round_sort_key)

    @property
    def rois(self) -> list[str]:
        """Discover and return all available regions of interest (ROIs) in the workspace.

        Uses regex pattern matching for robust directory name parsing. Scans workspace
        directories to extract ROI identifiers from directory names following the
        convention '{round}--{roi}' or '{process}--{roi}+{codebook}'.

        Returns:
            Sorted list of ROI identifiers (e.g., ['roi1', 'roi2', 'roi3'])

        Example:
            >>> ws = Workspace("/experiment")
            >>> ws.rois  # ['cortex', 'hippocampus', 'striatum']
        """

        rois_set = set()
        for path in [self.path, self.deconved]:
            if not path.exists():
                continue

            for p in path.iterdir():
                if not p.is_dir():
                    continue
                match = self.ROI_CODEBOOK_PATTERN.match(p.name)
                if match:
                    roi_name = match.group(1)
                    rois_set.add(roi_name)

        return sorted(rois_set)

    def resolve_rois(self, rois: Iterable[str] | None = None) -> list[str]:
        """Validate and normalize requested ROI identifiers.

        Args:
            rois: Optional iterable of ROI identifiers. If None, all available
                ROIs in the workspace are returned.

        Returns:
            Sorted list of ROI identifiers.

        Raises:
            ValueError: If any requested ROI does not exist in the workspace.
        """

        available = set(self.rois)
        if not rois:
            return sorted(available)

        requested = sorted({roi for roi in rois if roi})
        unknown = [roi for roi in requested if roi not in available]
        if unknown:
            raise ValueError(
                "Unknown ROI(s): {}. Available choices: {}".format(
                    ", ".join(unknown),
                    ", ".join(sorted(available)) or "none",
                )
            )
        return requested

    def registered_file_map(
        self, codebook: str, *, rois: Iterable[str] | None = None
    ) -> tuple[dict[str, list[Path]], list[str]]:
        """Return registered TIFF files grouped by ROI for a codebook.

        Args:
            codebook: Codebook identifier (stem without extension).
            rois: Optional iterable of ROI identifiers to restrict lookup.

        Returns:
            Tuple containing a mapping of ROI → list of TIFF paths and a list of
            ROI identifiers that were requested but have no registered output.
        """

        resolved_rois = self.resolve_rois(rois)
        mappings: dict[str, list[Path]] = {}
        missing: list[str] = []

        for roi in resolved_rois:
            directory = self.registered(roi, codebook)
            if directory.exists():
                mappings[roi] = sorted(directory.glob("reg-*.tif"))
            else:
                missing.append(roi)

        return mappings, missing

    @staticmethod
    def ensure_tiff_readable(path: Path) -> None:
        """Raise CorruptedTiffError if a TIFF file cannot be read."""

        try:
            # Silence tifffile warnings and stderr chatter while probing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with redirect_stderr(io.StringIO()):
                    with TiffFile(path) as tif:
                        tif.asarray()
        except Exception as exc:  # pragma: no cover - exercised by callers
            raise CorruptedTiffError(path, exc) from exc

    @property
    def deconved(self) -> Path:
        """Return path to deconvolved/processed data directory.

        Returns:
            Path to analysis/deconv directory containing processed images

        Example:
            >>> ws.deconved  # PosixPath('/experiment/analysis/deconv')
        """
        return self.path / "analysis" / "deconv"

    @overload
    def img(self, round_: str, roi: str, idx: int, *, read: Literal[False] = ...) -> Path: ...
    @overload
    def img(self, round_: str, roi: str, idx: int, *, read: Literal[True]) -> npt.NDArray[np.uint16]: ...
    def img(self, round_: str, roi: str, idx: int, *, read: bool = False):
        """Access deconvolved images by round, ROI, and index.

        Provides type-safe access to processed image files with optional direct loading.
        Uses overloaded signatures to ensure correct return type based on read parameter.

        Args:
            round_: Imaging round identifier (e.g., '1_9_17')
            roi: Region of interest identifier (e.g., 'cortex')
            idx: Image index (0-based)
            read: If True, load and return image data; if False, return path

        Returns:
            Path object if read=False, numpy array if read=True

        Example:
            >>> path = ws.img('1_9_17', 'cortex', 42)  # Returns Path
            >>> data = ws.img('1_9_17', 'cortex', 42, read=True)  # Returns ndarray
        """
        path = self.deconved / f"{round_}--{roi}/{round_}-{idx:04d}.tif"
        if read:
            return imread(path)
        return path

    def registered(self, roi: str, codebook: str) -> Path:
        """Return path to registration results directory.

        Args:
            roi: Region of interest identifier
            codebook: Codebook name used for registration

        Returns:
            Path to registered image directory

        Example:
            >>> ws.registered('cortex', 'codebook_v1')
            # PosixPath('/experiment/analysis/deconv/registered--cortex+codebook_v1')
        """
        return self.deconved / f"registered--{roi}+{codebook}"

    @overload
    def regimg(self, roi: str, codebook: str, idx: int, *, read: Literal[False] = ...) -> Path: ...
    @overload
    def regimg(self, roi: str, codebook: str, idx: int, *, read: Literal[True]) -> npt.NDArray[np.uint16]: ...
    def regimg(self, roi: str, codebook: str, idx: int, *, read: bool = False):
        """Access registered images by ROI, codebook, and index.

        Provides type-safe access to registration results with optional direct loading.
        Uses overloaded signatures to ensure correct return type based on read parameter.

        Args:
            roi: Region of interest identifier
            codebook: Codebook name used for registration
            idx: Image index (0-based)
            read: If True, load and return image data; if False, return path

        Returns:
            Path object if read=False, numpy array if read=True

        Example:
            >>> path = ws.regimg('cortex', 'codebook_v1', 42)  # Returns Path
            >>> data = ws.regimg('cortex', 'codebook_v1', 42, read=True)  # Returns ndarray
        """
        path = self.registered(roi, codebook) / f"reg-{idx:04d}.tif"
        if read:
            return imread(path)
        return path

    def stitch(self, roi: str, codebook: str | None = None) -> Path:
        """Return path to stitching results directory.

        Args:
            roi: Region of interest identifier
            codebook: Optional codebook name for registration-based stitching

        Returns:
            Path to stitched image directory

        Example:
            >>> ws.stitch('cortex')  # PosixPath('...deconv/stitch--cortex')
            >>> ws.stitch('cortex', 'cb_v1')  # PosixPath('...deconv/stitch--cortex+cb_v1')
        """
        if codebook is None:
            return self.deconved / f"stitch--{roi}"
        return self.deconved / f"stitch--{roi}+{codebook}"

    def tileconfig(self, roi: str) -> "TileConfiguration":
        """Load tile configuration for stitched images.

        Args:
            roi: Region of interest identifier

        Returns:
            TileConfiguration object with registered tile positions

        Raises:
            FileNotFoundError: If stitching has not been performed yet

        Example:
            >>> config = ws.tileconfig('cortex')
            >>> print(config.tiles)  # Access tile positions
        """
        try:
            return TileConfiguration.from_file(
                self.path / f"stitch--{roi}" / "TileConfiguration.registered.txt"
            )
        except FileNotFoundError:
            raise FileNotFoundError("Haven't stitch/registered yet. Run preprocess stitch register first.")

    def segment(self, roi: str, codebook: str) -> Path:
        """Return path to segmentation results directory.

        Args:
            roi: Region of interest identifier
            codebook: Codebook name used for segmentation

        Returns:
            Path to segmentation results directory

        Example:
            >>> ws.segment('cortex', 'codebook_v1')
            # PosixPath('/experiment/analysis/deconv/segment--cortex+codebook_v1')
        """
        return self.deconved / f"segment--{roi}+{codebook}"

    def opt(self, codebook: str) -> OptimizePath:
        """Return path to optimization results directory.

        Args:
            codebook: Codebook name used for optimization

        Returns:
            OptimizePath object for accessing optimization results

        Note:
            Real pattern verified: opt_{codebook} (underscore, not double-dash)

        Example:
            >>> ws.opt("ebe_tricycle_targets")
            # OptimizePath('/experiment/analysis/deconv/opt_ebe_tricycle_targets')
        """
        return OptimizePath(self.deconved / f"opt_{codebook}")


def get_metadata(file: Path):
    with TiffFile(file) as tif:
        try:
            meta = tif.shaped_metadata[0]  # type: ignore
        except KeyError:
            raise AttributeError("No metadata found.")
    return meta


def get_channels(file: Path):
    meta = get_metadata(file)
    waveform = json.loads(meta["waveform"])
    powers = waveform["params"]["powers"]
    return [name[-3:] for name in powers]
