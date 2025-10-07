import io
import re
import warnings
from contextlib import redirect_stderr, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, overload

import numpy as np
import numpy.typing as npt
from loguru import logger
from tifffile import TiffFile, imread
from tifffile import imwrite as tifffile_imwrite

from fishtools.io.codebook import Codebook
from fishtools.preprocess.tileconfig import TileConfiguration
from fishtools.utils.fs import download, get_file_name, set_cwd
from fishtools.utils.tiff import (
    get_channels as _ft_get_channels,
)
from fishtools.utils.tiff import (
    get_metadata as _ft_get_metadata,
)
from fishtools.utils.tiff import (
    normalize_channel_names,
    read_metadata_from_tif,
)


class CorruptedTiffError(RuntimeError):
    """Raised when a TIFF file cannot be read due to corruption."""

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"File {path} is corrupted. Please check the file.")
        self.path = path
        self.__cause__ = cause


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


@dataclass
class OptimizePath:
    path: Path

    @property
    def mse(self):
        return self.path / "mse.txt"

    @property
    def scaling_factor(self):
        return self.path / "global_scale.txt"


# Re-export Codebook for backward compatibility (imported above)


@dataclass
class Workspace:
    """FISH experiment workspace manager with verified directory structure.

    Provides standardized access to FISH data following the processing pipeline:
    Raw Data → Deconv → Register → Stitch → Analysis

    Directory Structure:
        workspace/
        ├── {round}--{roi}/                        # Raw imaging data
        │   ├── {round}-0001.tif
        │   └── {round}-0002.tif
        ├── stitch--{roi}/                         # ROI-level tile config directory
        │   └── TileConfiguration.registered.txt   # ImageJ Grid/Collection config
        └── analysis/
            ├── deconv/
            │   ├── {round}--{roi}/                # Deconvolved images
            │   ├── registered--{roi}+{codebook}/  # Registration results
            │   ├── stitch--{roi}/                 # ROI-level stitched outputs
            │   ├── stitch--{roi}+{codebook}/      # ROI+codebook stitched outputs
            │   ├── segment--{roi}+{codebook}/     # Segmentation results
            │   ├── shifts--{roi}+{codebook}/      # Registration shifts
            │   │   └── shifts-0001.json
            │   ├── fids--{roi}/                   # Fiducial markers
            │   │   └── fids-0001.tif
            │   └── opt_{codebook}/                # Optimization results
            │       ├── mse.txt
            │       └── global_scale.txt
            └── output/                            # Final analysis output

    CLI Pipeline & I/O:
        deconv:     {round}--{roi}/*.tif → analysis/deconv/{round}--{roi}/*.tif
        register:   analysis/deconv/{round}--{roi}/ → analysis/deconv/registered--{roi}+{codebook}/
        stitch:
            - Tile configuration is written/read at workspace_root/stitch--{roi}/
            - Stitched outputs: analysis/deconv/stitch--{roi}[+{codebook}]/
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

    _PLACEHOLDER_CHANNEL = re.compile(r"^channel_\d+$", re.IGNORECASE)

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
        oripath = path = Path(path).expanduser().resolve()
        stepped_up = 0
        while True:
            if stepped_up > 2:
                raise ValueError(f"Path {oripath} is not a valid FISH experiment workspace.")
            if any(p.suffix == ".DONE" for p in path.iterdir() if p.is_file()):
                break
            path = path.parent
            stepped_up += 1
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

    @staticmethod
    def _split_round_tokens(value: str) -> list[str]:
        """Tokenize a round/tile stem using repository naming conventions."""

        base = value
        for sep in ("--", "+"):
            if sep in base:
                base = base.split(sep, 1)[0]
        if "-" in base:
            base = base.split("-", 1)[0]
        tokens = [part for part in base.replace("-", "_").split("_") if part]
        return tokens

    def _first_deconv32_tile(self, round_name: str, roi: str | None = None) -> Path | None:
        base = self.path / "analysis" / "deconv32"
        if not base.exists():
            return None

        pattern = f"{round_name}--*"
        roi_filter = roi

        for roi_dir in sorted(p for p in base.glob(pattern) if p.is_dir()):
            current_roi = roi_dir.name.split("--", 1)[-1]
            if roi_filter and not current_roi.startswith(roi_filter):
                continue
            for tile in sorted(roi_dir.glob(f"{round_name}-*.tif")):
                if tile.suffix.lower() == ".tif":
                    return tile
        return None

    @staticmethod
    def _channel_names_from_metadata(tile: Path) -> list[str] | None:
        try:
            with TiffFile(tile) as tif:
                metadata = read_metadata_from_tif(tif)
                count: int | None = None
                try:
                    series = tif.series[0]
                except Exception:
                    series = None  # pragma: no cover - missing series
                axes = getattr(series, "axes", None) if series is not None else None
                shape = getattr(series, "shape", None) if series is not None else None

                if isinstance(axes, str) and shape:
                    axes_upper = axes.upper()
                    if "C" in axes_upper:
                        count = int(shape[axes_upper.index("C")])
                elif axes is None and shape:
                    if len(shape) == 3:
                        count = int(shape[0])
                    elif len(shape) == 4:
                        count = int(shape[1])

                if not count or count <= 0:
                    return None

                names = normalize_channel_names(count, metadata)
        except Exception:
            return None

        if not names:
            return None
        return list(names)

    def infer_channel_names(
        self,
        round_name: str,
        *,
        roi: str | None = None,
        prefer_metadata: bool = True,
    ) -> list[str] | None:
        """Attempt to infer ordered channel names for a round.

        Priority:
        1. Explicit TIFF metadata from the first float32 tile in analysis/deconv32
        2. Tokens parsed from the round name (e.g., wga_brdu → ["wga", "brdu"])
        3. Tokens parsed from the tile stem prior to the index suffix

        Placeholder names (channel_0, channel_1, …) are ignored.
        Returns None when no informative names can be resolved.
        """

        candidates: list[list[str]] = []
        tile = self._first_deconv32_tile(round_name, roi)

        if prefer_metadata and tile is not None:
            names = self._channel_names_from_metadata(tile)
            if names:
                candidates.append(names)

        tokens_round = self._split_round_tokens(round_name)
        if tokens_round:
            candidates.append(tokens_round)

        if tile is not None:
            tokens_tile = self._split_round_tokens(tile.stem)
            if tokens_tile:
                candidates.append(tokens_tile)

        for seq in candidates:
            filtered = [name for name in seq if not self._PLACEHOLDER_CHANNEL.fullmatch(name)]
            if filtered:
                return filtered

        return None

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

    def registered_codebooks(self, *, rois: Iterable[str] | None = None) -> list[str]:
        """Enumerate registered codebooks present in the workspace.

        Args:
            rois: Optional iterable of ROI identifiers to restrict the search.

        Returns:
            Sorted list of unique codebook identifiers discovered under
            ``analysis/deconv/registered--{roi}+{codebook}``.
        """

        resolved_rois = self.resolve_rois(rois)
        discovered: set[str] = set()

        for roi in resolved_rois:
            prefix = f"registered--{roi}+"
            for entry in self.deconved.glob(f"{prefix}*"):
                if not entry.is_dir():
                    continue
                _, _, suffix = entry.name.partition("+")
                if not suffix:
                    continue
                discovered.add(suffix)

        return sorted(discovered)

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
        """Return path to stitched output directory (ROI or ROI+codebook).

        Naming and location semantics:
        - Tile configuration (TileConfiguration.registered.txt) is ROI-specific and
          lives at the workspace root under ``stitch--{roi}/``.
        - Stitched outputs are ROI+codebook-specific and live under
          ``analysis/deconv/stitch--{roi}+{codebook}/``.

        This accessor returns the stitched output directory. Use
        :meth:`tileconfig_dir` or :meth:`tileconfig` to access the ROI-level
        tile configuration.

        Args:
            roi: Region of interest identifier
            codebook: Optional codebook name for registration-based stitching

        Returns:
            Path to stitched image/output directory

        Example:
            >>> ws.stitch('cortex')  # PosixPath('.../analysis/deconv/stitch--cortex')
            >>> ws.stitch('cortex', 'cb_v1')  # PosixPath('.../analysis/deconv/stitch--cortex+cb_v1')
        """
        if codebook is None:
            return self.deconved / f"stitch--{roi}"
        return self.deconved / f"stitch--{roi}+{codebook}"

    def fids(self, roi: str) -> Path:
        """Return path to fiducial marker directory for a given ROI."""

        return self.deconved / f"fids--{roi}"

    def fid(self, roi: str, idx: int | str) -> Path:
        """Return path to a fiducial TIFF for the specified ROI and tile index."""

        if isinstance(idx, int):
            suffix = f"{idx:04d}"
        else:
            idx_str = str(idx)
            suffix = f"{int(idx_str):04d}" if idx_str.isdigit() else idx_str
        return self.fids(roi) / f"fids-{suffix}.tif"

    def tile_positions_csv(self, roi: str, *, position_file: Path | None = None) -> Path:
        """Resolve the CSV containing tile positions for a given ROI."""

        if position_file is not None:
            return position_file

        candidates = [
            self.deconved / f"{roi}.csv",
            self.path / f"{roi}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Tile position CSV not found; provide --position_file or place "
            f"{roi}.csv in {self.deconved} or {self.path}."
        )

    def tileconfig_dir(self, roi: str) -> Path:
        """Return the ROI-level directory containing the TileConfiguration file.

        Note: This directory is separate from stitched outputs (which live
        under ``analysis/deconv/stitch--{roi}[+{codebook}]``).
        """
        return self.deconved / f"stitch--{roi}"

    def tileconfig(self, roi: str) -> "TileConfiguration":
        """Load the ROI-level TileConfiguration.

        Reads from ``<workspace_root>/stitch--{roi}/TileConfiguration.registered.txt``.
        For backward compatibility, if the file is not present at the workspace
        root, a secondary lookup is attempted under
        ``analysis/deconv/stitch--{roi}/``.

        Args:
            roi: Region of interest identifier

        Returns:
            TileConfiguration object with registered tile positions

        Raises:
            FileNotFoundError: If no TileConfiguration could be located

        Example:
            >>> config = ws.tileconfig('cortex')
            >>> print(config.tiles)  # Access tile positions
        """
        file = self.deconved / f"stitch--{roi}" / "TileConfiguration.registered.txt"
        if file.exists():
            return TileConfiguration.from_file(file)

        raise FileNotFoundError(
            f"No registered TileConfig found at {file}. Run preprocess stitch register first."
        )

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


def get_metadata(file: Path):  # re-export from utils.tiff
    return _ft_get_metadata(file)


def get_channels(file: Path):  # re-export from utils.tiff
    return _ft_get_channels(file)
