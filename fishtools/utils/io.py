import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import requests
from loguru import logger
from tifffile import TiffFile, imread
from tqdm.auto import tqdm

from fishtools.preprocess.tileconfig import TileConfiguration


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
    NUMERIC_SORT_PATTERN = re.compile(r"^(\d+)_")  # For numerical sorting of rounds

    path: Path

    def __str__(self) -> str:
        """Return string representation of workspace path."""
        return str(self.path)

    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        return f"Workspace({self.path})"

    def __init__(self, path: Path | str, deconved: bool = False) -> None:
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
        """Discover and return all available imaging rounds in the workspace.

        Uses regex pattern matching for robust directory name parsing. Scans workspace
        directories for round subdirectories following the naming convention '{round}--{roi}'.
        Automatically filters out processing directories and sorts numerically by round identifier.

        Returns:
            List of round identifiers (e.g., ['1_9_17', '2_10_18', '3_11_19'])

        Raises:
            ValueError: If no valid round directories are found

        Example:
            >>> ws = Workspace("/experiment")
            >>> ws.rounds  # ['1_9_17', '2_10_18', '3_11_19']
        """
        FORBIDDEN = {"10x", "analysis", "shifts", "stitch", "fid", "registered", "old", "basic"}
        path = self.path if not self.deconved.exists() else self.deconved

        rounds_set = set()
        for p in path.iterdir():
            if not p.is_dir():
                continue

            # Use regex to parse directory names
            match = self.ROUND_ROI_PATTERN.match(p.name)
            if match:
                round_name = match.group(1)
                # Filter out forbidden prefixes
                if not any(round_name.startswith(bad) for bad in FORBIDDEN):
                    rounds_set.add(round_name)

        if not rounds_set:
            raise ValueError(f"No round subdirectories found in {path}.")

        # Sort numerically if possible, otherwise alphabetically
        def sort_key(x: str) -> str:
            numeric_match = self.NUMERIC_SORT_PATTERN.match(x)
            if numeric_match:
                return f"{int(numeric_match.group(1)):02d}{x[len(numeric_match.group(1)) :]}"
            return x

        return sorted(rounds_set, key=sort_key)

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
        path = self.path if not self.deconved.exists() else self.deconved

        rois_set = set()
        for p in path.iterdir():
            if not p.is_dir():
                continue

            # Use regex to extract ROI from directory names
            match = self.ROI_CODEBOOK_PATTERN.match(p.name)
            if match:
                roi_name = match.group(1)
                rois_set.add(roi_name)

        return sorted(rois_set)

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
