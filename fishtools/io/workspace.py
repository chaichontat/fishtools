import io
import re
import warnings
from contextlib import redirect_stderr, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, overload

import numpy as np
import numpy.typing as npt
from tifffile import TiffFile, imread
from tifffile import imwrite as tifffile_imwrite

from fishtools.preprocess.tileconfig import TileConfiguration
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


@dataclass
class FiducialPaths:
    """Fiducial and fiducial-debug paths scoped to a single ROI.

    This helper centralizes the on-disk layout for fiducial artifacts under
    the deconvolved tree (analysis/deconv), including:

    - Fiducial thumbnails per ROI/tile index:
      ``<deconved>/fids--{roi}/fids-{idx:04d}.tif``
    - Debug stacks and overlays (when registration runs with --debug):
      ``<analysis>/output/fids_debug/{roi}/`` (PNG overlays) and
      ``<analysis>/output/fids_debug/{roi}/tifs/`` (multi-channel TIFF stacks).
    """

    deconved_root: Path
    roi: str

    def __init__(self, deconved_root: Path, roi: str) -> None:
        self.deconved_root = Path(deconved_root).resolve()
        self.roi = roi

    @property
    def fid_dir(self) -> Path:
        """Directory containing fiducial thumbnails for this ROI."""
        return self.deconved_root / f"fids--{self.roi}"

    def fid_tile(self, idx: int | str) -> Path:
        """Path to a fiducial TIFF for the specified tile index."""
        if isinstance(idx, int):
            suffix = f"{idx:04d}"
        else:
            idx_str = str(idx)
            suffix = f"{int(idx_str):04d}" if idx_str.isdigit() else idx_str
        return self.fid_dir / f"fids-{suffix}.tif"

    @property
    def debug_dir(self) -> Path:
        """Directory containing debug fiducial stacks and overlays."""
        return self.deconved_root.parent / "output" / "fids_debug" / self.roi


# Backward compatibility: codebook utilities are resolved elsewhere


@dataclass(frozen=True, slots=True)
class WorkspaceSpotlookOutput:
    """Spotlook-specific outputs under an output root."""

    root: Path

    @property
    def spots_final_dir(self) -> Path:
        return self.root / "spots_final"

    @property
    def threshold_selection_dir(self) -> Path:
        return self.root / "threshold_selection"

    @property
    def scree_final_dir(self) -> Path:
        return self.root / "scree_final"

    @property
    def contours_dir(self) -> Path:
        return self.root / "contours"

    @property
    def spots_contours_dir(self) -> Path:
        return self.root / "spots_contours"

    def combined_spots_png(self, codebook: str) -> Path:
        return self.spots_final_dir / f"spots_all--{codebook}.png"

    def combined_threshold_png(self, codebook: str) -> Path:
        return self.threshold_selection_dir / f"threshold_selection_all+{codebook}.png"

    def contours_png(self, roi: str, codebook: str) -> Path:
        return self.contours_dir / f"contours--{roi}+{codebook}.png"

    def spots_contours_png(self, roi: str, codebook: str) -> Path:
        return self.spots_contours_dir / f"spots_contours--{roi}+{codebook}.png"

    def threshold_selection_png(self, roi: str, codebook: str) -> Path:
        return self.threshold_selection_dir / f"threshold_selection--{roi}+{codebook}.png"


@dataclass(frozen=True, slots=True)
class WorkspaceOutput:
    """Path-like accessor for outputs under ``analysis/output``.

    The object forwards ``Path`` methods while also providing typed sub-accessors
    (e.g. ``ws.output.spotlook``) to avoid hardcoded path fragments in CLIs.
    """

    root: Path

    def __getattr__(self, name: str) -> Any:
        return getattr(self.root, name)

    def __truediv__(self, other: str | Path) -> Path:
        return self.root / other

    def __fspath__(self) -> str:
        return self.root.__fspath__()

    def __str__(self) -> str:
        return str(self.root)

    def __repr__(self) -> str:
        return f"WorkspaceOutput({self.root})"

    @property
    def parquets(self) -> Path:
        return self.root / "parquets"

    @property
    def stitch_layout(self) -> Path:
        return self.root / "stitch_layout"

    @property
    def spotlook(self) -> WorkspaceSpotlookOutput:
        return WorkspaceSpotlookOutput(self.root)


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

    ANALYSIS_DIRNAME = "analysis"
    TILECONFIG_REGISTERED_FILENAME = "TileConfiguration.registered.txt"

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
        if not path.exists():
            raise ValueError(f"Path {oripath} does not exist.")
        if not path.is_dir():
            raise ValueError(f"Path {oripath} is not a directory.")
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

    @property
    def analysis(self) -> Path:
        """Return path to analysis directory."""
        return self.path / self.ANALYSIS_DIRNAME

    @property
    def logs(self) -> Path:
        """Return path to workspace-scoped logs directory."""
        return self.analysis / "logs"

    @property
    def output(self) -> "WorkspaceOutput":
        """Return top-level analysis output accessor.

        This is the canonical location for ROI-level aggregated artifacts
        (e.g., per-ROI spots parquet files written by the spots pipeline).

        Example:
            >>> ws.output.root  # PosixPath('/experiment/analysis/output')
        """
        return WorkspaceOutput(self.analysis / "output")

    @property
    def stitch_layout(self) -> Path:
        """Return path to stitch layout plot directory."""
        return self.output / "stitch_layout"

    @property
    def parquets(self) -> Path:
        """Return path to parquets output directory."""
        return self.output / "parquets"

    def threshold_parquet(
        self, roi: str, codebook: str, *, raw: bool = False, output_dir: Path | None = None
    ) -> Path:
        """Return path to threshold-filtered spots parquet."""
        base = (output_dir / "parquets") if output_dir is not None else self.parquets
        suffix = ".raw.parquet" if raw else ".parquet"
        return base / f"{roi}+{codebook}{suffix}"

    @property
    def deconv32(self) -> Path:
        """Return path to float32 staging deconvolution directory."""
        return self.analysis / "deconv32"

    def deconv_scaling(self, round_: str | None = None) -> Path:
        """Return path to deconvolution scaling directory."""
        if round_ is not None:
            return self.analysis / "deconv_scaling" / f"{round_}.txt"
        return self.analysis / "deconv_scaling"

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

        if "_bleach" in rounds_set:
            rounds_set.remove("_bleach")

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
                    # Discard everything after second '--' (e.g., 'roi1--shifted-1_9_17' → 'roi1')
                    if "--" in roi_name:
                        roi_name = roi_name.split("--")[0]
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

    @property
    def chromatic(self) -> Path:
        """Return path to chromatic correction resources directory.

        Stores per-workspace chromatic correction profiles such as ``560to650.txt`` and
        ``560to750.txt`` used by registration.
        """

        return self.deconved / "chromatic"

    def deconv_round_dir(self, round_name: str, roi: str) -> Path:
        """Return path to a deconvolved round/ROI directory under analysis/deconv."""

        return self.deconved / f"{round_name}--{roi}"

    def deconv_repaired_dir(self, round_name: str, roi: str) -> Path:
        """Return path to a repaired deconvolved round/ROI directory under analysis/deconv."""

        return self.deconved / f"{round_name}--{roi}--repaired"

    def shifts(self, roi: str, codebook: str | None = None) -> Path:
        """Return path to the shifts directory for a ROI (optionally codebook-scoped)."""

        if codebook is None:
            return self.deconved / f"shifts--{roi}"
        return self.deconved / f"shifts--{roi}+{codebook}"

    def shift_json(self, roi: str, codebook: str, idx: int) -> Path:
        """Return path to a per-tile shifts JSON file."""

        return self.shifts(roi, codebook) / f"shifts-{idx:04d}.json"

    def coarse_shifts_json(self, roi: str) -> Path:
        """Return path to the coarse shifts JSON produced by fix-shifts."""

        return self.shifts(roi) / "coarse_shifts.json"

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

    def registered_fids(self, roi: str, codebook: str) -> Path:
        """Return the `_fids` directory under a registration output directory."""

        return self.registered(roi, codebook) / "_fids"

    def decoded_dir(self, roi: str, codebook: str) -> Path:
        """Return the decoded output directory under a registration output directory."""

        return self.registered(roi, codebook) / f"decoded-{codebook}"

    def decoded_spots_parquet(self, roi: str, codebook: str) -> Path:
        """Return the decoded spots parquet path under the registered output tree."""

        return self.decoded_dir(roi, codebook) / "spots.parquet"

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
          lives under ``analysis/deconv/stitch--{roi}/``.
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

    def stitch_shifted(self, roi: str, round_name: str) -> Path:
        """Return path to a coarse-shifted stitch directory for a ROI and round name."""

        return self.deconved / f"stitch--{roi}--shifted-{round_name}"

    @staticmethod
    def sanitize_codebook_name(codebook: str) -> str:
        """Normalize a codebook label for filesystem-friendly paths.

        Replaces hyphens and spaces with underscores to match how other
        parts of the pipeline name ROI-level parquet files.
        """
        return codebook.replace("-", "_").replace(" ", "_")

    def spots_parquet(self, roi: str, codebook: str, *, must_exist: bool = False) -> Path:
        """Resolve or suggest the ROI-level spots parquet path.

        Search order (first existing is returned):
        1) analysis/output/parquets/{roi}+{sanitize(codebook)}.parquet
        2) analysis/output/parquets/{roi}+{codebook}.parquet
        3) analysis/output/{roi}+{sanitize(codebook)}.parquet
        4) analysis/output/{roi}+{codebook}.parquet
        5) analysis/deconv/{roi}+{sanitize(codebook)}.parquet
        6) analysis/deconv/{roi}+{codebook}.parquet

        When ``must_exist`` is False and no candidates exist, returns the
        preferred default path under ``analysis/output/parquets`` with the
        sanitized codebook.
        """
        cb_s = self.sanitize_codebook_name(codebook)
        candidates = [
            self.parquets / f"{roi}+{cb_s}.parquet",
            self.parquets / f"{roi}+{codebook}.parquet",
            self.output / f"{roi}+{cb_s}.parquet",
            self.output / f"{roi}+{codebook}.parquet",
            self.deconved / f"{roi}+{cb_s}.parquet",
            self.deconved / f"{roi}+{codebook}.parquet",
        ]
        for p in candidates:
            if p.exists():
                return p
        if must_exist:
            searched = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(
                f"Spots parquet not found for ROI '{roi}', codebook '{codebook}'. Searched: {searched}"
            )
        return candidates[0]

    def fids(self, roi: str) -> Path:
        """Return path to fiducial marker directory for a given ROI."""

        return FiducialPaths(self.deconved, roi).fid_dir

    def fid(self, roi: str, idx: int | str) -> Path:
        """Return path to a fiducial TIFF for the specified ROI and tile index."""

        return FiducialPaths(self.deconved, roi).fid_tile(idx)

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

        Contract:
        - ROI-level TileConfiguration directories live under the deconvolved tree:
          ``<workspace>/analysis/deconv/stitch--{roi}/``.
        """
        return self.deconved / f"stitch--{roi}"

    def tileconfig_registered_txt(self, roi: str) -> Path:
        """Return the canonical ROI-level TileConfiguration file path.

        Contract:
        - ``<workspace_root>/analysis/deconv/stitch--{roi}/TileConfiguration.registered.txt``.
        """
        return self.tileconfig_dir(roi) / self.TILECONFIG_REGISTERED_FILENAME

    def tileconfig(self, roi: str) -> "TileConfiguration":
        """Load the ROI-level TileConfiguration.

        Reads from the canonical location:

        - ``<workspace_root>/analysis/deconv/stitch--{roi}/TileConfiguration.registered.txt``.

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
        path = self.tileconfig_registered_txt(roi)
        if not path.exists():
            raise FileNotFoundError(
                f"No registered TileConfig found at {path}. Run `preprocess stitch register` first."
            )
        return TileConfiguration.from_file(path)

    def fields_dir(self, codebook: str) -> Path:
        """Return path to the illumination field store directory for a codebook."""

        slug = self.sanitize_codebook_name(codebook)
        return self.deconved / f"fields+{slug}"

    def field_zarr(self, roi: str, codebook: str) -> Path:
        """Return path to a TCYX illumination field Zarr store."""

        slug = self.sanitize_codebook_name(codebook)
        return self.fields_dir(slug) / f"field--{roi}+{slug}.zarr"

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

    @staticmethod
    def is_bit_round(round_name: str) -> bool:
        """Determine if a round is a bit round (vs. named/non-bit round).

        Bit rounds typically have numeric prefixes like "1_9_17" indicating
        channel indices, while non-bit rounds have descriptive names like
        "dapi", "polyA", etc.

        Args:
            round_name: The round identifier to check

        Returns:
            True if the round is a bit round, False otherwise

        Example:
            >>> Workspace.is_bit_round("1_9_17")  # True
            >>> Workspace.is_bit_round("dapi")    # False
        """
        if not round_name:
            return False
        first_token = round_name.split("_")[0]
        return first_token.isdigit()

    def pending_deconv_tasks(
        self, include_nonbit: bool = False
    ) -> list[tuple[str, str, Literal["run", "prepare", "run_u16"]]]:
        """Discover pending deconvolution tasks for this workspace.

        Scans the workspace to identify which rounds and ROIs still need
        deconvolution processing. Bit rounds require a "run" task that produces
        float32 output. Non-bit rounds (when include_nonbit=True) require a
        "prepare" task followed by "run_u16" tasks for each ROI.

        Args:
            include_nonbit: If True, include prepare/run_u16 tasks for non-bit rounds

        Returns:
            List of (round_name, roi, task_type) tuples where task_type is:
            - "run": Deconvolution needed for a bit round
            - "prepare": Scaling computation needed for a non-bit round
            - "run_u16": U16 deconvolution needed for a non-bit round

        Example:
            >>> ws.pending_deconv_tasks()
            [('1_9_17', 'roi1', 'run'), ('1_9_17', 'roi2', 'run')]
            >>> ws.pending_deconv_tasks(include_nonbit=True)
            [('1_9_17', 'roi1', 'run'), ('dapi', '', 'prepare')]
        """
        tasks: list[tuple[str, str, Literal["run", "prepare", "run_u16"]]] = []

        for round_name in self.rounds:
            is_bit = self.is_bit_round(round_name)

            if is_bit:
                # Bit rounds: check each ROI for deconvolved output
                for roi in self.rois:
                    src_dir = self.path / f"{round_name}--{roi}"
                    if not src_dir.exists():
                        continue

                    # Count source tiles
                    src_tiles = list(src_dir.glob(f"{round_name}-*.tif"))
                    if not src_tiles:
                        continue

                    # Check if deconvolved output exists
                    deconv_dir = self.deconved / f"{round_name}--{roi}"
                    if deconv_dir.exists():
                        deconv_tiles = list(deconv_dir.glob(f"{round_name}-*.tif"))
                        if len(deconv_tiles) >= len(src_tiles):
                            continue

                    # Deconvolution needed
                    tasks.append((round_name, roi, "run"))

            elif include_nonbit:
                # Non-bit rounds: check if prepare/precompute has been done
                scaling_file = self.deconv_scaling(round_name)
                if not scaling_file.exists():
                    # Need to run prepare + precompute
                    tasks.append((round_name, "", "prepare"))
                else:
                    # Scaling exists, now check run_u16 for each ROI with source tiles
                    for roi in self.rois:
                        src_dir = self.path / f"{round_name}--{roi}"
                        if not src_dir.exists():
                            continue

                        src_tiles = list(src_dir.glob(f"{round_name}-*.tif"))
                        if not src_tiles:
                            continue

                        # Check if u16 deconvolved output exists
                        deconv_dir = self.deconved / f"{round_name}--{roi}"
                        if deconv_dir.exists():
                            deconv_tiles = list(deconv_dir.glob(f"{round_name}-*.tif"))
                            if len(deconv_tiles) >= len(src_tiles):
                                continue

                        # U16 deconvolution needed
                        tasks.append((round_name, roi, "run_u16"))

        return tasks


def get_metadata(file: Path):  # re-export from utils.tiff
    return _ft_get_metadata(file)


def get_channels(file: Path):  # re-export from utils.tiff
    return _ft_get_channels(file)
