# %%
import csv
import json
import logging
import pickle
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import rich_click as click
import toml
import tifffile  # noqa: F401
from basicpy import BaSiC
from loguru import logger
from pydantic import ValidationError
from scipy import ndimage
from scipy.ndimage import shift
from tifffile import TiffFile

from fishtools.preprocess.chromatic import Affine, apply_dense_xy_displacement_field
from fishtools.preprocess.config import (
    Config,
    Fiducial,
    FiducialDetailedConfig,
    NumpyEncoder,
    RegisterConfig,
    default_register_config,
    resolve_data_path,
)
from fishtools.preprocess.deconv.helpers import scale_deconv
from fishtools.preprocess.downsample import gpu_downsample_xy
from fishtools.preprocess.fiducial import (
    FiducialAlignmentStats,
    Shifts,
    align_fiducials_with_stats,
    shifts_from_anchor_roi,
)
from fishtools.gpu.memory import release_all as gpu_release_all
from fishtools.utils.io import FiducialPaths, Workspace, safe_imwrite
from fishtools.utils.logging import setup_cli_logging
from fishtools.utils.pretty_print import progress_bar_threadpool, run_subprocess_streaming

# =============================================================================
# COORDINATE CONVENTIONS
# =============================================================================
# Shifts dictionary: [dx, dy] where dx is column displacement (x-axis) and dy is row
# displacement (y-axis).
# SciPy: scipy.ndimage.shift expects [row_shift, col_shift] = [dy, dx], so we swap
# when calling it.
# Sign: +dx/+dy means the target image is displaced +x/+y relative to the reference;
# to correct (resample the target into the reference frame) apply the negative shift.
# ITK: SimpleITK returns translation parameters [tx, ty]; we convert to our [dx, dy]
# convention (and still swap to [dy, dx] for ndimage.shift).

FORBIDDEN_PREFIXES = ["10x", "registered", "shifts", "fids"]

if TYPE_CHECKING:
    pass


# %%

DATA = resolve_data_path()


# %%


def _silence_matplotlib_debug_logs() -> None:
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def _load_chromatic_affines(
    ws: Workspace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    As: dict[str, np.ndarray] = {}
    ats: dict[str, np.ndarray] = {}
    meta: dict[str, dict[str, Any]] = {}

    for λ in ["650", "750"]:
        filename = f"560to{λ}.txt"
        path = ws.output.chromatic / filename
        if not path.exists():
            raise click.ClickException(
                f"Missing chromatic correction file {path}. "
                "Stage chromatic corrections under analysis/output/chromatic (set registration.chromatic_path via --config)."
            )
        a_ = np.loadtxt(path)
        A = np.zeros((3, 3), dtype=np.float64)
        A[:2, :2] = a_[:4].reshape(2, 2)
        t = np.zeros(3, dtype=np.float64)
        t[:2] = a_[-2:]

        A[2] = [0, 0, 1]
        A[:, 2] = [0, 0, 1]
        t[2] = 0
        As[λ] = A
        ats[λ] = t
        meta[λ] = {
            "source": str(path),
            "A": [[float(x) for x in row] for row in A[:2, :2]],
            "t": [float(x) for x in t[:2]],
        }

    meta["ref"] = {"channel": "560"}
    return As, ats, meta


def _load_optional_chromatic_displacement_fields(
    ws: Workspace,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, Any]]]:
    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for λ in ["650", "750"]:
        path = ws.output.chromatic / f"560to{λ}_field.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as loaded:
            if "dx_dense" not in loaded or "dy_dense" not in loaded or "meta_json" not in loaded:
                raise click.ClickException(
                    f"Invalid chromatic field file {path} (expected keys: dx_dense, dy_dense, meta_json)."
                )
            dx = np.asarray(loaded["dx_dense"], dtype=np.float32)
            dy = np.asarray(loaded["dy_dense"], dtype=np.float32)
            if dx.shape != dy.shape or dx.ndim != 2:
                raise click.ClickException(
                    f"Invalid chromatic field file {path} (dx_dense/dy_dense must be 2D and same shape)."
                )
            fields[λ] = (dx, dy)
            meta_json = str(loaded["meta_json"].item())
            mj = json.loads(meta_json)
            if not isinstance(mj, dict) or "kind" not in mj:
                raise click.ClickException(f"Invalid chromatic field file {path} (meta_json missing 'kind').")
            meta[λ] = {"source": str(path), "kind": str(mj["kind"])}
    return fields, meta


_chromatic_logged = False


def _log_chromatic_matrices_once(As: dict[str, np.ndarray], ats: dict[str, np.ndarray]) -> None:
    """Log chromatic affine matrices once per process (debug mode only)."""
    global _chromatic_logged
    if _chromatic_logged:
        return
    _chromatic_logged = True
    for λ in ["650", "750"]:
        logger.debug(f"Chromatic 560→{λ}: A=\n{As[λ][:2,:2]}\n  t={ats[λ][:2]}")


def spillover_correction(spillee: np.ndarray, spiller: np.ndarray, corr: float):
    scaled = spiller * corr
    return np.where(spillee >= scaled, spillee - scaled, 0)


def parse_nofids(
    nofids: dict[str, np.ndarray],
    shifts: dict[str, np.ndarray],
    channels: dict[str, str],
):
    """Converts nofids into bits and perform shift correction.

    Args:
        nofids: {name: zcyx images}
        shifts: {name: shift vector}
        channels: channel {bit: name, must be 488, 560, 650, 750}


    Raises:
        ValueError: Duplicate bit names.

    Returns:
        out: {bit: zyx images}
        out_shift: {bit: shift vector}
        bit_name_mapping: {bit: (name, idx)} for deconv scaling.
    """

    out: dict[str, Annotated[np.ndarray, "z,y,x"]] = {}
    out_shift: dict[str, Annotated[np.ndarray, "shifts"]] = {}
    bit_name_mapping: dict[str, tuple[str, int]] = {}

    for name, img in nofids.items():
        curr_bits = name.split("-")[0].split("_")
        assert img.shape[1] == len(curr_bits)

        for i, bit in enumerate(curr_bits):
            bit_name_mapping[bit] = (name, i)

            if bit in out:
                raise ValueError(f"Duplicated bit {bit} in {name}")
            out[bit] = img[:, i]  # sliced.max(0, keepdims=True) if max_proj else sliced

        # cs = {str(channels[bit]): bit for bit in bits}
        # if "560" in cs and "647" in cs:
        #     out[cs["560"]] = spillover_correction(out[cs["560"]], out[cs["647"]], 0.22)

        # if "647" in cs and "750" in cs:
        #     out[cs["647"]] = spillover_correction(out[cs["647"]], out[cs["750"]], 0.05)

    for name, shift_ in shifts.items():
        curr_bits = name.split("-")[0].split("_")
        for i, bit in enumerate(curr_bits):
            out_shift[bit] = shift_

    return out, out_shift, bit_name_mapping


def _run_child_cli(
    argv: list[str],
    *,
    check: bool = True,
) -> CompletedProcess[str]:
    return run_subprocess_streaming(argv, check=check)


def _parse_repaired_option(value: str | None) -> set[str] | None:
    """Convert --repaired CLI input into a normalized set of round names."""

    if value is None:
        return None

    rounds = {token.strip() for token in value.split(",") if token.strip()}
    return rounds or None


def _load_outlier_tiles_from_shifts_metrics(
    csv_path: Path,
    *,
    only_median_gt: float,
) -> list[int]:
    if not csv_path.exists():
        raise click.ClickException(f"Missing shifts metrics CSV at {csv_path}")

    outliers: set[int] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = {name for name in ("tile", "L2") if name not in fieldnames}
        if missing:
            raise click.ClickException(f"Invalid shifts metrics CSV at {csv_path}: missing columns {sorted(missing)}")

        for row in reader:
            try:
                tile = int(str(row.get("tile", "")).strip())
                l2 = float(str(row.get("L2", "")).strip())
            except (TypeError, ValueError) as exc:
                raise click.ClickException(f"Invalid shifts metrics row in {csv_path}: {row}") from exc
            if l2 > only_median_gt:
                outliers.add(tile)

    return sorted(outliers)


def _load_low_corr_tiles_from_shifts_metrics(
    csv_path: Path,
    *,
    only_corr_lt: float,
) -> list[int]:
    if not csv_path.exists():
        raise click.ClickException(f"Missing shifts metrics CSV at {csv_path}")

    selected: set[int] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = {name for name in ("tile", "correlation") if name not in fieldnames}
        if missing:
            raise click.ClickException(f"Invalid shifts metrics CSV at {csv_path}: missing columns {sorted(missing)}")

        for row in reader:
            try:
                tile = int(str(row.get("tile", "")).strip())
                corr = float(str(row.get("correlation", "")).strip())
            except (TypeError, ValueError) as exc:
                raise click.ClickException(f"Invalid shifts metrics row in {csv_path}: {row}") from exc
            if corr < only_corr_lt:
                selected.add(tile)

    return sorted(selected)


def _write_shifts_json(
    ws: Workspace,
    *,
    roi: str,
    codebook: str,
    idx: int,
    payload: bytes,
) -> None:
    deconv_dir = ws.shifts(roi, codebook)
    deconv_dir.mkdir(exist_ok=True, parents=True)
    (deconv_dir / f"shifts-{idx:04d}.json").write_bytes(payload)


def _copy_codebook_to_workspace(cli_path: Path, codebook_path: Path) -> Path:
    workspace = Workspace(cli_path)
    codebooks_dir = workspace.deconved / "codebooks"
    codebooks_dir.mkdir(parents=True, exist_ok=True)

    source = Path(codebook_path).resolve(strict=True)
    destination = codebooks_dir / source.name
    try:
        if destination.resolve(strict=True) == source:
            return destination
    except FileNotFoundError:
        pass

    shutil.copy2(source, destination)
    return destination


def _copy_chromatic_corrections_to_output(cli_path: Path, chromatic_dir: Path) -> None:
    workspace = Workspace(cli_path)
    output_chromatic_dir = workspace.output.chromatic
    output_chromatic_dir.mkdir(parents=True, exist_ok=True)

    source_dir = chromatic_dir.resolve(strict=True)
    required = ("560to650.txt", "560to750.txt")
    for filename in required:
        source = source_dir / filename
        if not source.exists():
            raise click.ClickException(f"Chromatic directory {source_dir} is missing required file {source.name}")

    copied: dict[str, dict[str, str]] = {}
    optional = ("560to650_field.npz", "560to750_field.npz")
    for filename in required + optional:
        source = source_dir / filename
        if not source.exists():
            continue
        destination = output_chromatic_dir / filename
        shutil.copy2(source, destination)
        copied[filename] = {"source": str(source), "destination": str(destination)}

    provenance: dict[str, Any] = {"source_dir": str(source_dir), "files": copied}
    (output_chromatic_dir / "chromatic_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def sort_key(x: str | tuple[str, np.ndarray]) -> str:
    """Key function for dict items / strings by numerical order.

    Args:
        x: Bit name string or dict.items() tuple with bit name as first element.

    Returns:
        Key for sorting.
    """
    name = x if isinstance(x, str) else x[0]
    try:
        return f"{int(name):02d}"
    except ValueError:
        return name


def apply_deconv_scaling(
    img: np.ndarray,
    *,
    idx: int,
    orig_name: str,
    global_deconv_scaling: np.ndarray,
    metadata: dict[str, Any],
    debug: bool,
) -> np.ndarray:
    """Apply deconvolution rescaling unless input is already prenormalized."""

    if metadata.get("prenormalized"):
        if debug:
            logger.debug(f"Skipping deconvolution scaling for {orig_name}: metadata prenormalized flag set.")
        return img

    return scale_deconv(
        img,
        idx,
        name=orig_name,
        global_deconv_scaling=global_deconv_scaling,
        metadata=metadata,
        debug=debug,
    )


def _apply_priors_to_fids(
    fids: dict[str, np.ndarray],
    fid_raw_images: dict[str, np.ndarray] | None,
    *,
    priors: dict[str, tuple[float, float]] | None,
    anchor_roi: Path | None,
    idx: int,
) -> dict[str, str]:
    """Shift fiducials in-place using priors. Returns prior key → fid name mapping."""
    if priors is None or anchor_roi is not None:
        return {}

    prior_mapping: dict[str, str] = {}
    for name, sh in priors.items():
        for file in fids:
            if file.startswith(name):
                fids[file] = shift(fids[file], [sh[1], sh[0]], order=1)
                if fid_raw_images is not None and file in fid_raw_images:
                    fid_raw_images[file] = shift(fid_raw_images[file], [sh[1], sh[0]], order=1)
                prior_mapping[name] = file
                break
        else:
            raise ValueError(
                f"{idx}: Searched {list(fids.keys())}. Could not find file that starts with {name} for prior shift."
            )

    return prior_mapping


def _add_priors_to_shifts(
    shifts: dict[str, np.ndarray],
    *,
    priors: dict[str, tuple[float, float]] | None,
    prior_mapping: dict[str, str],
    anchor_roi: Path | None,
):
    """Add prior offsets back onto solved shift vectors."""
    if priors is None or anchor_roi is not None:
        return

    for name, sh in priors.items():
        mapped = prior_mapping.get(name)
        if mapped is None:
            continue
        shifts[mapped][0] += sh[0]
        shifts[mapped][1] += sh[1]


def _parse_priors_file(path: Path | None) -> dict[str, tuple[float, float]] | None:
    """Load explicit priors from a JSON or CSV file."""

    if path is None:
        return None

    raw = path.read_text().strip()
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        pairs: dict[str, tuple[float, float]] = {}
        reader = csv.reader(raw.splitlines())
        for row in reader:
            if not row:
                continue
            # Allow optional header row
            if len(row) == 3 and row[0].strip().lower() == "round":
                continue
            if len(row) != 3:
                raise ValueError("CSV priors must have exactly three columns: round,dx,dy.")
            name, dx_str, dy_str = row
            pairs[name.strip()] = (float(dx_str), float(dy_str))
        return pairs or None

    if not isinstance(data, dict):
        raise ValueError("--priors JSON must map round → [dx, dy].")

    parsed: dict[str, tuple[float, float]] = {}
    for key, value in data.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Each prior entry must be a two-element array of [dx, dy].")
        parsed[str(key)] = (float(value[0]), float(value[1]))
    return parsed or None


def _load_register_config_from_json(path: Path) -> RegisterConfig:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise click.ClickException(f"--config file is empty: {path}")

    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in --config file {path}: {e}") from e

    if not isinstance(payload, dict):
        raise click.ClickException(f"--config JSON must be an object/dict, got {type(payload).__name__}: {path}")

    def _deep_merge(base: Any, patch: Any) -> Any:
        if isinstance(base, dict) and isinstance(patch, dict):
            merged = dict(base)
            for k, v in patch.items():
                merged[k] = _deep_merge(merged.get(k), v) if k in merged else v
            return merged
        return patch

    config_keys = set(Config.model_fields.keys())
    is_config_shape = bool(set(payload.keys()) & config_keys)

    if is_config_shape:
        base = Config()
        merged_dict = _deep_merge(base.model_dump(mode="python"), payload)
        try:
            full_config = Config.model_validate(merged_dict)
        except ValidationError as e:
            raise click.ClickException(f"Invalid --config JSON at {path}:\n{e}") from e
        config = full_config.registration
    else:
        base = default_register_config()
        merged_dict = _deep_merge(base.model_dump(mode="python"), payload)
        try:
            config = RegisterConfig.model_validate(merged_dict)
        except ValidationError as e:
            raise click.ClickException(f"Invalid --config JSON at {path}:\n{e}") from e

    chromatic_path = config.chromatic_path
    if not chromatic_path.is_absolute():
        chromatic_path = (path.parent / chromatic_path).resolve()

    anchor_roi = config.fiducial.anchor_roi
    if anchor_roi is not None and not anchor_roi.is_absolute():
        anchor_roi = (path.parent / anchor_roi).resolve()

    fiducial = config.fiducial.model_copy(update={"anchor_roi": anchor_roi})
    return config.model_copy(update={"chromatic_path": chromatic_path, "fiducial": fiducial})


def _apply_cli_overrides_to_registration_config(
    loaded: RegisterConfig,
    *,
    reference: str | None = None,
    threshold: float | None = None,
    fwhm: float | None = None,
    use_fft: bool | None = None,
    use_itk: bool | None = None,
    anchors: Path | None = None,
    use_brightest: int | None = None,
    offset_brightest: int | None = None,
    allow_large_shifts: bool | None = None,
    n_fids: int | None = None,
    priors: dict[str, tuple[float, float]] | None = None,
) -> RegisterConfig:
    """Apply non-None CLI overrides onto a loaded RegisterConfig."""

    fid_updates: dict[str, Any] = {}
    detailed_updates: dict[str, Any] = {}

    if threshold is not None:
        fid_updates["threshold"] = threshold
    if fwhm is not None:
        fid_updates["fwhm"] = fwhm
    if use_fft is not None:
        fid_updates["use_fft"] = use_fft
    if use_itk is not None:
        fid_updates["use_itk"] = use_itk
    if anchors is not None:
        fid_updates["anchor_roi"] = anchors
    if n_fids is not None:
        fid_updates["n_fids"] = n_fids
    if priors is not None:
        fid_updates["priors"] = priors

    if use_brightest is not None:
        detailed_updates["use_brightest"] = max(use_brightest, 0)
    if offset_brightest is not None:
        detailed_updates["offset_brightest"] = max(offset_brightest, 0)
    if allow_large_shifts is not None:
        detailed_updates["allow_large_shifts"] = allow_large_shifts

    detailed = (
        loaded.fiducial.detailed.model_copy(update=detailed_updates)
        if detailed_updates
        else loaded.fiducial.detailed
    )
    fiducial = (
        loaded.fiducial.model_copy(update={**fid_updates, "detailed": detailed})
        if (fid_updates or detailed_updates)
        else loaded.fiducial
    )

    updates: dict[str, Any] = {"fiducial": fiducial}
    if reference is not None:
        updates["reference"] = reference

    return loaded.model_copy(update=updates)


def _build_register_config(
    *,
    reference: str,
    threshold: float,
    fwhm: float,
    use_fft: bool,
    use_itk: bool,
    use_brightest: int,
    offset_brightest: int,
    allow_large_shifts: bool,
    n_fids: int,
    chromatic_dir: Path,
    anchor_roi: Path | None = None,
    priors: dict[str, tuple[float, float]] | None = None,
) -> RegisterConfig:
    """Create a RegisterConfig with consistent defaults for fiducial alignment."""

    resolved_priors = priors if priors is not None else {}
    return RegisterConfig(
        chromatic_path=chromatic_dir.resolve(),
        fiducial=Fiducial(
            use_fft=use_fft,
            use_itk=use_itk,
            fwhm=fwhm,
            threshold=threshold,
            priors=resolved_priors,
            overrides={},
            anchor_roi=anchor_roi,
            n_fids=n_fids,
            detailed=FiducialDetailedConfig(
                use_brightest=max(use_brightest, 0),
                offset_brightest=max(offset_brightest, 0),
                allow_large_shifts=allow_large_shifts,
            ),
        ),
        reference=reference,
        downsample=1,
        crop=40,
        slices=slice(None),
        reduce_bit_depth=0,
        discards=None,
    )


def _save_debug_overlay(
    debug_dir: Path,
    roi: str,
    idx: int,
    reference_name: str,
    shifted: dict[str, np.ndarray],
    *,
    codebook_name: str,
) -> Path | None:
    """Save red-green overlay figure: reference round (green) vs shifted rounds (red)."""
    import matplotlib.pyplot as plt

    ref = shifted[reference_name]

    def norm_pct(arr: np.ndarray) -> np.ndarray:
        p10, p99999 = np.percentile(arr, [10, 99.999])
        scaled = np.clip((arr - p10) / (p99999 - p10 + 1e-8), 0, 1)
        return np.sqrt(scaled)

    ref_norm = (norm_pct(ref) * 255).astype(np.uint8)

    other_names = [k for k in sorted(shifted) if k != reference_name]
    if not other_names:
        return None

    n_panels = len(other_names)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=200, facecolor="black", squeeze=False
    )
    axes = axes.ravel()
    blue = np.zeros_like(ref_norm)

    for ax, name in zip(axes, other_names):
        img_norm = (norm_pct(shifted[name]) * 255).astype(np.uint8)
        rgb = np.stack([img_norm, ref_norm, blue], axis=-1)
        ax.imshow(rgb)
        ax.set_title(name, fontsize=10, color="white")
        ax.set_facecolor("black")
        ax.axis("off")

    for ax in axes[n_panels:]:
        ax.set_facecolor("black")
        ax.axis("off")

    fig.suptitle(f"{roi}-{idx:04d} (green=ref:{reference_name})", fontsize=12, color="white")
    fig.tight_layout()
    overlay_path = debug_dir / f"{roi}+{codebook_name}-{idx:04d}-overlay.png"
    fig.savefig(overlay_path, facecolor="black")
    plt.close(fig)
    return overlay_path


def _debug_fid_paths(path: Path, roi: str, idx: int, codebook_name: str) -> tuple[Path, str, str]:
    paths = FiducialPaths(path, roi)
    debug_dir = paths.debug_dir
    fids_name = f"{roi}+{codebook_name}-{idx:04d}.tif"
    shifted_name = f"{roi}+{codebook_name}-shifted-{idx:04d}.tif"
    return debug_dir, fids_name, shifted_name


@dataclass
class Image:
    name: str
    idx: int
    nofid: np.ndarray
    fid: np.ndarray
    fid_raw: np.ndarray
    bits: list[str]
    powers: dict[str, float]
    metadata: dict[str, Any]
    global_deconv_scaling: np.ndarray | None
    basic: Callable[[], dict[str, BaSiC] | None]

    CHANNELS = [f"ilm{n}" for n in ["405", "488", "560", "650", "750"]]

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        discards: dict[str, list[str]] | None = None,
        n_fids: int = 1,
        load_nofid: bool = True,
    ):
        ws = Workspace(path.parent.parent)
        stem = path.stem
        name, idx = stem.split("-")
        bits = name.split("_")
        if discards is None:
            discards = {}

        to_discard_idxs = []
        for k, v in discards.items():
            if k in bits and name in v:
                logger.debug(f"Discarding {k} (index {bits.index(k)}) from {stem}.")
                to_discard_idxs.append(bits.index(k))

        with TiffFile(path) as tif:
            try:
                if load_nofid:
                    img = tif.asarray()
                    fid_stack = np.atleast_3d(img[-n_fids:])
                else:
                    if n_fids <= 0:
                        raise ValueError("n_fids must be > 0")
                    if len(tif.pages) < n_fids:
                        raise ValueError(
                            f"{path}: expected at least {n_fids} frames, got {len(tif.pages)}"
                        )
                    fid_stack = np.stack([p.asarray() for p in tif.pages[-n_fids:]], axis=0)
                    img = None
                try:
                    metadata = tif.shaped_metadata[0]  # type: ignore
                except IndexError:
                    metadata = tif.imagej_metadata
                # tifffile throws IndexError if the file is truncated
            except IndexError as e:
                raise Exception(f"File {path} is corrupted. Please check the file.") from e
            assert metadata is not None

        try:
            waveform = (
                metadata["waveform"]
                if isinstance(metadata["waveform"], dict)
                else json.loads(metadata["waveform"])
            )
        except KeyError:
            waveform = toml.load(path.with_name(f"{path.name.split('-')[0]}.toml"))

        counts = {key: sum(waveform[key]["sequence"]) for key in cls.CHANNELS}

        # To remove ilm from, say, ilm405.

        powers = {
            key[3:]: waveform[key]["power"]
            for key in cls.CHANNELS
            if (key == "ilm405" and counts[key] > n_fids) or (key != "ilm405" and counts[key])
        }

        if waveform.get("params"):
            powers = waveform["params"]["powers"]

        if len(powers) != len(bits):
            raise ValueError(f"{path}: Expected {len(bits)} channels, got {len(powers)}")

        prenormalized = metadata.get("prenormalized", False)
        if not prenormalized:
            try:
                global_deconv_scaling = (
                    np.loadtxt(ws.deconv_scaling(name)).astype(np.float32).reshape((2, -1))
                )
            except FileNotFoundError:
                raise ValueError(f"No deconv_scaling found for {name} and prenormalized not set.")
        else:
            global_deconv_scaling = None

        if img is None:
            fids_raw = np.atleast_3d(fid_stack).max(axis=0)
            nofid = np.empty((0, 0, 0, 0), dtype=np.uint16)
        else:
            tile_h, tile_w = img.shape[-2], img.shape[-1]
            nofid = img[:-n_fids].reshape(-1, len(powers), tile_h, tile_w)
            fids_raw = np.atleast_3d(fid_stack).max(axis=0)

        if to_discard_idxs:
            _bits = name.split("_")
            power_keys = list(powers.keys())
            for _idx_d in to_discard_idxs:
                _bits.pop(_idx_d)
                del powers[power_keys[_idx_d]]
            name = "_".join(_bits)
            keeps = list(sorted(set(range(len(bits))) - set(to_discard_idxs)))
            if img is not None:
                nofid = nofid[:, keeps]
            bits = [bits[i] for i in keeps]
            global_deconv_scaling = (
                global_deconv_scaling[:, keeps] if global_deconv_scaling is not None else None
            )
            if img is not None:
                assert len(_bits) == nofid.shape[1]

        path_basic = path.parent.parent / "basic" / f"{name}.pkl"
        if path_basic.exists():

            def b():
                basic = pickle.loads(path_basic.read_bytes())
                return dict(zip(bits, basic.values()))

            basic = b
        else:
            # raise Exception(f"No basic template found at {path_basic}")
            basic = lambda: None

        return cls(
            name=name,
            idx=int(idx),
            nofid=nofid,
            fid=cls.loG_fids(fids_raw),
            fid_raw=fids_raw,
            bits=bits,
            powers=powers,
            metadata=metadata,
            global_deconv_scaling=global_deconv_scaling,
            basic=basic,
        )

    @staticmethod
    def loG_fids(fid: np.ndarray):
        if len(fid.shape) == 3:
            fid = fid.max(axis=0)

        temp = -ndimage.gaussian_laplace(fid.astype(np.float32, copy=False), sigma=3)  # type: ignore
        temp = temp.astype(np.float32, copy=False)
        temp -= temp.min()
        percs = np.percentile(temp, [1, 99.99]).astype(np.float32, copy=False)

        if percs[1] - percs[0] == 0:
            raise ValueError("Uniform image")
        temp = (temp - percs[0]) / (percs[1] - percs[0])
        return temp.astype(np.float32, copy=False)


def run_fiducial(
    path: Path,
    fids: dict[str, np.ndarray],
    codebook_name: str,
    config: Config,
    *,
    roi: str,
    idx: int,
    reference: str,
    debug: bool,
    prior_only: bool = False,
    no_priors: bool = False,
    fids_raw: dict[str, np.ndarray],
    max_iters: int = 5,
):
    ws = Workspace(path)
    prior_mapping: dict[str, str] = {}

    shifts_existing_count: int | None = None
    if (
        config.registration.fiducial.anchor_roi is None  # Skip priors when using anchor ROI
        and len(shifts_existing := sorted(ws.shifts(roi, codebook_name).glob("*.json"))) > 10
        and not no_priors
        and not config.registration.fiducial.priors
    ):
        shifts_existing_count = len(shifts_existing)
        _priors: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for shift_path in shifts_existing:
            try:
                shift_dicts = Shifts.validate_json(shift_path.read_text())
            except ValidationError as e:
                logger.warning(f"Error decoding {shift_path} {e}. Skipping.")
                continue
            for name, shift_dict in shift_dicts.items():
                if shift_dict.residual < 0.3:
                    _priors[name].append(shift_dict.shifts)
        logger.debug(f"Using priors from {len(shifts_existing)} existing shifts.")
        config.registration.fiducial.priors = {
            name: tuple(np.median(np.array(shifts), axis=0)) for name, shifts in _priors.items()
        }
        logger.debug(config.registration.fiducial.priors)
    elif config.registration.fiducial.anchor_roi is None and debug:
        shifts_existing_count = len(sorted(ws.shifts(roi, codebook_name).glob("*.json")))

    prior_mapping |= _apply_priors_to_fids(
        fids,
        fids_raw,
        priors=config.registration.fiducial.priors,
        anchor_roi=config.registration.fiducial.anchor_roi,
        idx=idx,
    )

    if prior_only:
        if no_priors:
            raise click.ClickException("--use-prior-only cannot be combined with --no-priors.")
        if config.registration.fiducial.anchor_roi is not None:
            raise click.ClickException("--use-prior-only cannot be combined with --anchors.")
        if not config.registration.fiducial.priors:
            raise click.ClickException(
                "--use-prior-only requires prior shifts. Provide explicit priors or generate them "
                "by running registration on enough tiles."
            )

    if debug and config.registration.fiducial.anchor_roi is None:
        if no_priors:
            logger.debug("Priors: disabled via --no-priors")
        elif not config.registration.fiducial.priors:
            count_msg = (
                f"{shifts_existing_count}" if shifts_existing_count is not None else "unknown"
            )
            logger.debug(f"Priors: none (existing shifts files={count_msg}; need >10 to auto-derive)")
        else:
            lines: list[str] = []
            for name, sh in sorted(config.registration.fiducial.priors.items()):
                mapped = prior_mapping.get(name)
                if mapped is None:
                    continue
                dx, dy = float(sh[0]), float(sh[1])
                lines.append(f"{name} -> {mapped}: dx={dx:.3f}, dy={dy:.3f}")
            if lines:
                logger.debug("Applied priors:\n" + "\n".join(f"  {line}" for line in lines))
            else:
                logger.debug("Priors: configured but none applied (no matching fid keys)")

    if config.registration.fiducial.overrides is not None:
        for name, sh in config.registration.fiducial.overrides.items():
            for file in fids:
                if file.startswith(name):
                    logger.info(f"Overriding shift for {name} to {sh}")
                    fids[file] = shift(fids[file], [sh[1], sh[0]], order=1)
                    prior_mapping[name] = file
                    break
            else:
                raise ValueError(f"Could not find file that starts with {name} for override shift.")

    # Write reference fiducial
    fid_paths = FiducialPaths(path, roi)
    fid_path = fid_paths.fid_dir
    fid_path.mkdir(exist_ok=True)
    crop = config.registration.crop
    if crop:
        fid_img = fids[reference][crop:-crop, crop:-crop]
    else:
        fid_img = fids[reference]
    safe_imwrite(
        fid_paths.fid_tile(idx),
        fid_img,
        compression=22610,
        compressionargs={"level": 0.65},
        metadata={"axes": "YX", "key": [reference]},
    )

    # Use a deterministic ordering for all multi-channel fiducial stacks
    ordered_keys = sorted(fids.keys())

    _fids_path = ws.registered_fids(roi, codebook_name)
    _fids_path.mkdir(exist_ok=True, parents=True)

    safe_imwrite(
        _fids_path / f"_fids-{idx:04d}.tif",
        np.stack([fids_raw[k] for k in ordered_keys]),
        compression=22610,
        compressionargs={"level": 0.65},
        metadata={"axes": "CYX", "key": ordered_keys},
    )

    if config.registration.fiducial.anchor_roi is not None:
        logger.info(f"Using anchor points from {config.registration.fiducial.anchor_roi}")
        anchor_shifts = shifts_from_anchor_roi(
            config.registration.fiducial.anchor_roi,
            reference,
            ordered_keys,
        )
        shifts = {k: anchor_shifts.get(k, np.array([0.0, 0.0])) for k in fids}
        residuals = {k: 0.0 for k in fids}
        stats: dict[str, FiducialAlignmentStats | None] = {k: None for k in fids}
    elif prior_only:
        shifts = {k: np.array([0.0, 0.0]) for k in fids}
        residuals = {k: 0.0 for k in fids}
        stats = {k: None for k in fids}
    else:
        shifts, residuals, stats = align_fiducials_with_stats(
            fids,
            reference=reference,
            debug=debug,
            max_iters=max_iters,
            threshold_sigma=config.registration.fiducial.threshold,
            fwhm=config.registration.fiducial.fwhm,
            use_fft=config.registration.fiducial.use_fft,
            use_itk=config.registration.fiducial.use_itk,
            use_brightest=config.registration.fiducial.detailed.use_brightest,
            detailed_config=config.registration.fiducial.detailed,
        )

        assert shifts  # type: ignore
        assert residuals  # type: ignore

    shifted = {k: shift(fid, [shifts[k][1], shifts[k][0]]) for k, fid in fids.items()}

    if debug:
        debug_dir, fids_name, shifted_name = _debug_fid_paths(path, roi, idx, codebook_name)
        debug_dir.mkdir(exist_ok=True, parents=True)
        tifs_dir = debug_dir / "tifs"
        tifs_dir.mkdir(exist_ok=True, parents=True)

        use_raw = config.registration.fiducial.use_fft or config.registration.fiducial.use_itk
        debug_fids = (
            {k: Image.loG_fids(fids_raw[k]) for k in ordered_keys}
            if use_raw
            else {k: fids[k] for k in ordered_keys}
        )
        debug_shifted = {
            k: shift(debug_fids[k], [shifts[k][1], shifts[k][0]]) for k in ordered_keys
        }
        # Keep channel order stable for QC tooling.
        safe_imwrite(
            tifs_dir / fids_name,
            np.stack([debug_fids[k] for k in ordered_keys]),
            compression=22610,
            compressionargs={"level": 0.65},
            metadata={"axes": "CYX", "key": ordered_keys},
        )
        safe_imwrite(
            tifs_dir / shifted_name,
            np.stack([debug_shifted[k] for k in ordered_keys]),
            compression=22610,
            compressionargs={"level": 0.65},
            metadata={"axes": "CYX", "key": ordered_keys},
        )
        overlay_path = _save_debug_overlay(
            debug_dir, roi, idx, reference, debug_shifted, codebook_name=codebook_name
        )
        if overlay_path is not None:
            logger.info(f"fids_debug overlay image: {overlay_path.resolve()}")

    # Add priors to final shifts (skip when using anchor ROIs - anchor points are absolute)
    _add_priors_to_shifts(
        shifts,
        priors=config.registration.fiducial.priors,
        prior_mapping=prior_mapping,
        anchor_roi=config.registration.fiducial.anchor_roi,
    )

    _fid_ref = fids[reference][500:-500:2, 500:-500:2].flatten()
    validated = Shifts.validate_python(
        {
            k: {
                "shifts": (shifts[k][0], shifts[k][1]),
                "residual": residuals[k],
                "corr": 1.0
                if reference == k
                else np.corrcoef(shifted[k][500:-500:2, 500:-500:2].flatten(), _fid_ref)[0, 1],
                "iterations": (stats.get(k).iterations if stats.get(k) is not None else None),
                "final_fwhm": (stats.get(k).final_fwhm if stats.get(k) is not None else None),
                "final_threshold": (stats.get(k).final_threshold if stats.get(k) is not None else None),
                "n_spots": (stats.get(k).n_spots if stats.get(k) is not None else None),
                "mode": (stats.get(k).mode if stats.get(k) is not None else None),
                "algorithm": (stats.get(k).algorithm if stats.get(k) is not None else None),
            }
            for k in fids
        }
    )
    jsoned = Shifts.dump_json(validated)
    _write_shifts_json(ws, roi=roi, codebook=codebook_name, idx=idx, payload=jsoned)
    logger.debug({k: f"{r.corr:03f}" for k, r in validated.items()})
    return shifts


def _run(
    path: Path,
    roi: str,
    idx: int,
    *,
    codebook: str | Path,
    reference: str,
    config: Config,
    debug: bool = False,
    overwrite: bool = False,
    no_priors: bool = False,
    prior_only: bool = False,
    repaired_rounds: set[str] | None = None,
    max_iters: int = 5,
    use_shifts_from: str | None = None,
):
    logger.info("Starting")
    codebook_name = Path(codebook).stem
    ws = Workspace(path)
    out_path = ws.registered(roi, codebook_name)
    reg_file = ws.regimg(roi, codebook_name, idx)

    if not overwrite and reg_file.exists():
        logger.info(f"Skipping {idx}")
        return

    out_path.mkdir(exist_ok=True, parents=True)

    # path_prevfids = out_path / "_fids" / f"_fids-{idx:04d}.tif"
    # if path_prevfids.exists():
    #     with TiffFile(path_prevfids) as tif:
    #         _fids = tif.asarray()
    #         fids = {k: v for k, v in zip(tif.shaped_metadata[0]["key"], _fids)}
    #         del _fids
    #     shifts = run_fiducial(
    #         path,
    #         fids,
    #         Path(codebook).stem,
    #         config,
    #         roi=roi,
    #         reference=reference,
    #         debug=debug,
    #         idx=idx,
    #         no_priors=no_priors,
    #     )
    #     del fids
    # else:
    shifts: dict[str, np.ndarray] = {}

    # Load shifts from another codebook if requested
    if use_shifts_from is not None:
        logger.info(f"Loading shifts from codebook '{use_shifts_from}', skipping fiducial registration.")
        shifts = _load_shifts_from_codebook(ws, roi=roi, source_codebook=use_shifts_from, idx=idx)

    cb = json.loads(Path(codebook).read_text())
    codebook_bits = {str(bit) for bit in chain.from_iterable(cb.values())}

    # Build list of round directories, redirecting repaired rounds to --repaired folders
    repaired_rounds = set(repaired_rounds or set())
    roi_dirs = []
    used_repaired: set[str] = set()
    repaired_paths: dict[str, Path] = {}
    for p in Path(path).glob(f"*--{roi}"):
        if not p.is_dir():
            continue
        if any(p.name.startswith(bad) for bad in FORBIDDEN_PREFIXES + (config.exclude or [])):
            continue
        # Check if this round should use repaired folder
        round_name = p.name.split("--")[0]
        if round_name in repaired_rounds:
            repaired_path = ws.deconv_repaired_dir(round_name, roi)
            if repaired_path.exists():
                roi_dirs.append(repaired_path)
                logger.info(f"Using repaired folder for round {round_name}: {repaired_path}")
                used_repaired.add(round_name)
                repaired_paths[round_name] = repaired_path
            else:
                raise FileNotFoundError(f"Repaired folder for round {round_name} not found: {repaired_path}")
        else:
            roi_dirs.append(p)

    if repaired_rounds:
        unused_repaired = repaired_rounds - used_repaired
        if unused_repaired:
            missing_list = ", ".join(sorted(unused_repaired))
            logger.warning(f"Requested --repaired rounds not found for ROI {roi}: {missing_list}")

    available_bits = {bit for p in roi_dirs for bit in p.name.split("--")[0].split("_") if bit}

    missing_bits = sorted(bit for bit in codebook_bits if bit not in available_bits)
    available_display = ", ".join(sorted(available_bits)) or "none"
    if missing_bits:
        raise ValueError(
            f"Missing codebook bits for ROI {roi}: {', '.join(missing_bits)} (available: {available_display})"
        )

    reference_bits = set(reference.split("_"))
    if repaired_paths:
        for round_name, repaired_path in repaired_paths.items():
            round_bits = set(round_name.split("_"))
            if not (round_bits & (codebook_bits | reference_bits)):
                continue
            if not any(repaired_path.glob(f"*-{idx:04d}.tif")):
                raise FileNotFoundError(
                    f"Repaired folder for round {round_name} lacks index {idx:04d} in {repaired_path}. "
                    "Remove --repaired for this round or repair/regenerate the missing file."
                )
    folders = {
        p for p in roi_dirs if set(p.name.split("--")[0].split("_")) & (codebook_bits | reference_bits)
    }

    files = [
        file
        for file in chain.from_iterable(p.glob(f"*-{idx:04d}.tif") for p in folders)
        if not any(file.parent.name.startswith(bad) for bad in FORBIDDEN_PREFIXES + (config.exclude or []))
    ]
    if not files:
        raise FileNotFoundError(f"No files found in {path} with index {idx}")

    if not shifts:
        # Fiducial-only pass to keep memory low while solving shifts.
        fid_imgs = {
            img.name: img
            for img in (
                Image.from_file(
                    file,
                    discards=config.registration and config.registration.discards,
                    n_fids=config.registration.fiducial.n_fids,
                    load_nofid=False,
                )
                for file in files
            )
        }

        # Use raw fiducials for FFT/ITK (ITK applies its own preprocessing)
        # Use LoG-processed fiducials for spot-based alignment
        use_fft = config.registration.fiducial.use_fft
        use_itk = config.registration.fiducial.use_itk
        use_raw = use_fft or use_itk
        fid_raw_images = {name: img.fid_raw.astype(np.float32, copy=False) for name, img in fid_imgs.items()}
        fid_images = fid_raw_images.copy() if use_raw else {name: img.fid for name, img in fid_imgs.items()}
        if reference not in fid_raw_images:
            logger.info(f"Loading reference fiducial {reference} from previous run.")
            raw_ref = _load_reference_fid_from_previous_run(
                ws,
                roi=roi,
                reference=reference,
                idx=idx,
                prefer_codebook=codebook_name,
            )
            fid_raw_images[reference] = raw_ref
            fid_images[reference] = raw_ref if use_raw else Image.loG_fids(raw_ref)

        # if not use_fft and not use_itk:
        shifts = run_fiducial(
            path,
            fid_images,
            codebook_name,
            config,
            roi=roi,
            reference=reference,
            debug=debug,
            idx=idx,
            prior_only=prior_only,
            no_priors=no_priors,
            fids_raw=fid_raw_images,
            max_iters=max_iters,
        )
        del fid_imgs, fid_images, fid_raw_images

    # Full pass: load the full stacks after shifts are known.
    _imgs: list[Image] = [
        Image.from_file(
            file,
            discards=config.registration and config.registration.discards,
            n_fids=config.registration.fiducial.n_fids,
        )
        for file in files
    ]
    imgs = {img.name: img for img in _imgs}
    del _imgs

    logger.debug(f"{len(imgs)} files: {list(imgs)}")
    assert imgs

    for _img in imgs.values():
        del _img.fid, _img.fid_raw

    # Remove reference if not in codebook since we're done with fiducials.
    if reference in imgs and not (set(reference.split("_")) & codebook_bits):
        del imgs[reference]

    channels: dict[str, str] = {}
    for img in imgs.values():
        channels |= dict(zip(img.bits, [p[-3:] for p in img.powers]))

    assert all(v.isdigit() for v in channels.values())

    logger.debug(f"Channels: {channels}")

    # Split into individual bits.
    # Spillover correction, max projection
    nofids = {name: img.nofid for name, img in imgs.items()}
    bits, bits_shifted, bit_name_mapping = parse_nofids(nofids, shifts, channels)

    del nofids
    for _img in imgs.values():
        del _img.nofid

    # if debug:
    #     for name, img in bits.items():
    #         logger.info(f"{name}: {img.max()}")

    def collapse_z(
        img: np.ndarray,
        slices: list[tuple[int | None, int | None]] | slice | None,
    ) -> np.ndarray:
        if isinstance(slices, list):
            return np.stack([img[slice(*sl)].max(axis=0) for sl in slices])
        if slices is None:
            slices = slice(None)
        return img[slices]

    bits_in_output = sorted(codebook_bits & set(bits))
    missing_channels = [bit for bit in bits_in_output if bit not in channels]
    if missing_channels:
        raise KeyError(missing_channels[0])

    keys = sorted(bits_in_output, key=sort_key)
    out: np.ndarray | None = None

    required_chromatic = ("560to650.txt", "560to750.txt")
    if any(not (ws.output.chromatic / filename).exists() for filename in required_chromatic):
        source_dir = config.registration.chromatic_path.resolve()
        output_dir = ws.output.chromatic.resolve()
        if source_dir != output_dir:
            _copy_chromatic_corrections_to_output(path, source_dir)

    As, ats, chromatic_meta = _load_chromatic_affines(ws)
    chromatic_fields, chromatic_field_meta = _load_optional_chromatic_displacement_fields(ws)
    for k, v in chromatic_field_meta.items():
        chromatic_meta.setdefault(k, {})
        chromatic_meta[k]["field"] = v
    if debug:
        _log_chromatic_matrices_once(As, ats)
    affine = Affine(As=As, ats=ats)

    needs_gpu_cleanup = config.registration.downsample > 1
    try:
        ref_set = False
        for i, bit in enumerate(keys):
            bit = str(bit)
            img = bits[bit]
            del bits[bit]
            c = str(channels[bit])
            # Deconvolution scaling
            orig_name, orig_idx = bit_name_mapping[bit]
            img = collapse_z(img, config.registration.slices).astype(np.float32, copy=False)
            metadata = imgs[orig_name].metadata

            if not metadata.get("prenormalized"):
                scaling = imgs[orig_name].global_deconv_scaling
                assert scaling is not None
                img = apply_deconv_scaling(
                    img,
                    idx=orig_idx,
                    orig_name=orig_name,
                    global_deconv_scaling=scaling,
                    metadata=metadata,
                    debug=debug,
                )
                del scaling

            if not ref_set:
                # Need to put this here because of shape change during collapse_z.
                affine.ref_image = img
                ref_set = True

            # Within-tile alignment. Chromatic corrections.
            logger.debug(f"{bit}: before affine channel={c}, shiftpx={-bits_shifted[bit]}")
            img = affine(img, channel=c, shiftpx=-bits_shifted[bit], debug=debug)
            if c in chromatic_fields:
                dx_dense, dy_dense = chromatic_fields[c]
                img = apply_dense_xy_displacement_field(img, dx_dense=dx_dense, dy_dense=dy_dense)
            crop = config.registration.crop
            downsample = config.registration.downsample
            if crop:
                img = img[:, crop:-crop, crop:-crop]

            if downsample > 1:
                transformed_img = gpu_downsample_xy(
                    img,
                    crop=0,
                    factor=downsample,
                    clip_range=(0, 65534),
                    output_dtype=np.uint16,
                )
            else:
                transformed_img = np.clip(img, 0, 65534).astype(np.uint16)

            if out is None:
                out = np.empty(
                    (
                        transformed_img.shape[0],
                        len(keys),
                        transformed_img.shape[1],
                        transformed_img.shape[2],
                    ),
                    dtype=np.uint16,
                )
            elif transformed_img.shape != out[:, 0].shape:
                raise ValueError(
                    f"Transformed images have different shapes: expected {out[:, 0].shape}, got {transformed_img.shape}"
                )
            out[:, i] = transformed_img
            logger.debug(f"Transformed {bit}: max={img.max()}, min={img.min()}")
            logger.debug(f"Finished transforming bit={bit} ({i + 1}/{len(keys)}) for idx={idx:04d}")
    finally:
        if needs_gpu_cleanup:
            gpu_release_all()

    # Drop any unrequested bits (e.g., extras present in round files) to release backing arrays.
    bits.clear()

    if out is None:
        raise ValueError("No images were transformed.")
    # out[0, -1] = fids[reference][crop:-crop:downsample, crop:-crop:downsample]
    logger.debug(str([f"{i}: {k}" for i, k in enumerate(keys, 1)]))

    # (path / "down2").mkdir(exist_ok=True)

    safe_imwrite(
        out_path / f"reg-{idx:04d}.tif",
        out,
        compression=22610,
        compressionargs={"level": 0.75},
        metadata={
            "key": keys,
            "axes": "ZCYX",
            "shifts": json.dumps(shifts, cls=NumpyEncoder),
            "config": json.dumps(config.model_dump(mode="json"), cls=NumpyEncoder),
            "chromatic": json.dumps(chromatic_meta),
        },
    )
    logger.info(f"Wrote reg-{idx:04d}.tif to {out_path}")

    # for i in range(0, len(out), 3):
    #     (path / "down2" / str(i)).mkdir(exist_ok=True, parents=True)
    #     tifffile.imwrite(
    #         path / "down2" / str(i) / f"{i:03d}-{idx:04d}.tif",
    #         out[i : i + 3] >> config.reduce_bit_depth,
    #         compression=22610,
    #         compressionargs={"level": 0.9},
    #         metadata={"axes": "CYX", "channels": ",".join(keys[i : i + 3])},
    #         imagej=True,
    #     )


def get_rois(path: Path, roi: str):
    rois = (
        {r.name.split("+")[0].split("--")[1] for r in path.glob("*--*") if r.is_dir()}
        if roi == "*"
        else [roi]
    )
    return {r for r in rois if r and r != "*"}


def _build_register_run_argv(
    *,
    path: Path,
    idx: int,
    codebook: Path,
    config: Path,
    fwhm: float | None,
    threshold: float | None,
    reference: str | None,
    roi: str,
    max_iters: int,
    overwrite: bool,
    debug: bool,
    repaired: str | None,
    use_fft: bool | None,
    use_itk: bool | None,
    use_brightest: int | None,
    offset_brightest: int | None,
    allow_large_shifts: bool | None,
    use_shifts_from: str | None,
    use_prior_only: bool = False,
) -> list[str]:
    argv = [
        "preprocess",
        "register",
        "run",
        str(path),
        str(idx),
        f"--codebook={codebook}",
        f"--config={config}",
        *([f"--fwhm={fwhm}"] if fwhm is not None else []),
        *([f"--threshold={threshold}"] if threshold is not None else []),
        *(["--reference", reference] if reference is not None else []),
        f"--roi={roi}",
        f"--max-iters={max_iters}",
        *(["--overwrite"] if overwrite else []),
        *(["--debug"] if debug else []),
        *([f"--repaired={repaired}"] if repaired else []),
        *(["--use-fft"] if use_fft else []),
        *(["--use-itk"] if use_itk else []),
        *([f"--use-brightest={use_brightest}"] if use_brightest is not None and use_brightest > 0 else []),
        *(
            [f"--offset-brightest={offset_brightest}"]
            if offset_brightest is not None and offset_brightest > 0
            else []
        ),
        *(["--allow-large-shifts"] if allow_large_shifts else []),
        *([f"--use-shifts-from={use_shifts_from}"] if use_shifts_from else []),
        *(["--use-prior-only"] if use_prior_only else []),
    ]
    return argv


@click.group()
def register(): ...


@register.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path, resolve_path=True))
@click.argument("idx", type=int)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path, resolve_path=True),
    default=None,
    help=(
        "Path to a JSON RegisterConfig (or Config containing 'registration'). "
        "Defaults to <workspace>/analysis/deconv/config.json then <workspace>/config.json when present. "
        "Command-line flags override config values."
    ),
)
@click.option("--codebook", type=click.Path(exists=True, file_okay=True, path_type=Path))
@click.option("--roi", type=str, default="*")
@click.option("--reference", "-r", type=str, default=None, show_default="from config")
@click.option("--debug", is_flag=True)
@click.option("--threshold", type=float, default=None, show_default="from config")
@click.option("--fwhm", type=float, default=None, show_default="from config")
@click.option("--overwrite", is_flag=True)
@click.option("--no-priors", is_flag=True)
@click.option(
    "--use-prior-only",
    is_flag=True,
    help="Skip automatic registration and use only the prior shifts (after shifting fiducials, residual shift is 0,0).",
)
@click.option(
    "--use-fft",
    is_flag=True,
    default=None,
    help="Use FFT phase correlation instead of spot-based matching.",
)
@click.option(
    "--use-itk",
    is_flag=True,
    default=None,
    help="Use SimpleITK gradient descent for alignment (more robust for low-contrast).",
)
@click.option("--anchors", type=click.Path(exists=True, path_type=Path), help="Path to ImageJ RoiSet.zip with anchor points. Skips automatic registration.")
@click.option(
    "--use-brightest",
    type=int,
    default=None,
    show_default="from config",
    help="If >0, use only the N brightest fiducial spots per image for spot-based alignment.",
)
@click.option(
    "--offset-brightest",
    type=click.IntRange(min=0),
    default=None,
    show_default="from config",
    help="Skip the first N brightest fiducial spots before applying --use-brightest.",
)
@click.option(
    "--allow-large-shifts",
    is_flag=True,
    default=None,
    help="Accept shifts larger than the configured threshold instead of raising DriftTooLarge.",
)
@click.option(
    "--repaired",
    type=str,
    default=None,
    help="Comma-separated round names to use from --repaired folders (e.g., '1_9_17,2_10_18')",
)
@click.option(
    "--max-iters",
    type=int,
    default=5,
    show_default=True,
    help="Maximum iterations for spot-based drift refinement.",
)
@click.option(
    "--use-shifts-from",
    type=str,
    default=None,
    help="Codebook name to copy shifts from, skipping fiducial registration entirely.",
)
def run(
    path: Path,
    idx: int,
    config_path: Path | None,
    codebook: Path,
    roi: str | None,
    debug: bool = False,
    reference: str | None = None,
    overwrite: bool = False,
    threshold: float | None = None,
    fwhm: float | None = None,
    no_priors: bool = False,
    use_prior_only: bool = False,
    use_fft: bool | None = None,
    use_itk: bool | None = None,
    anchors: Path | None = None,
    use_brightest: int | None = None,
    offset_brightest: int | None = None,
    allow_large_shifts: bool | None = None,
    repaired: str | None = None,
    max_iters: int = 5,
    use_shifts_from: str | None = None,
):
    """Preprocess image sets before spot calling.

    Args:
        path: Workspace path.
        idx: Index of the image to process.
        roi: ROI to work on.
        debug: More logs and write fids. Defaults to False.
        overwrite: Defaults to False.
        threshold: σ above median to call fiducial spots. Defaults to the value in --config.
        fwhm: FWHM for the Gaussian spot detector. Defaults to the value in --config.
    """
    ws = Workspace(path)
    if config_path is None:
        config_path = ws.config_json()
        if config_path is None:
            raise click.ClickException(
                "No config.json found. Provide --config or create "
                "'analysis/deconv/config.json' (preferred) or 'config.json' at the workspace root."
            )

    rois = get_rois(path, roi)
    codebook = _copy_codebook_to_workspace(path, codebook)
    codebook_name = codebook.stem

    loaded_registration = _load_register_config_from_json(config_path)
    _copy_chromatic_corrections_to_output(path, loaded_registration.chromatic_path)
    staged_chromatic_dir = ws.output.chromatic

    reference_effective = reference or loaded_registration.reference
    registration = _apply_cli_overrides_to_registration_config(
        loaded_registration,
        reference=reference_effective,
        threshold=threshold,
        fwhm=fwhm,
        use_fft=use_fft,
        use_itk=use_itk,
        anchors=anchors,
        use_brightest=use_brightest,
        offset_brightest=offset_brightest,
        allow_large_shifts=allow_large_shifts,
    ).model_copy(update={"chromatic_path": staged_chromatic_dir})

    if registration.fiducial.detailed.offset_brightest > 0 and registration.fiducial.detailed.use_brightest <= 0:
        raise click.ClickException("--offset-brightest requires --use-brightest > 0.")

    if use_prior_only and no_priors:
        raise click.ClickException("--use-prior-only cannot be combined with --no-priors.")
    if use_prior_only and anchors is not None:
        raise click.ClickException("--use-prior-only cannot be combined with --anchors.")
    if use_prior_only and use_shifts_from is not None:
        raise click.ClickException("--use-prior-only cannot be combined with --use-shifts-from.")

    for roi in rois:
        reg_file = ws.regimg(roi, codebook_name, idx)
        if not overwrite and reg_file.exists():
            logger.info(f"Skipping {idx}: registration already present at {reg_file}")
            continue

        log_file_tag = f"{roi}+{codebook_name}+{idx:04d}"
        setup_cli_logging(
            path,
            component="preprocess.register.run",
            file=log_file_tag,
            idx=idx,
            debug=debug,
            extra={"roi": roi, "codebook": codebook_name},
        )
        _silence_matplotlib_debug_logs()

        # Parse repaired rounds
        repaired_rounds = _parse_repaired_option(repaired)

        _run(
            path,
            roi,
            idx,
            codebook=codebook,
            debug=debug,
            reference=reference_effective,
            no_priors=no_priors,
            prior_only=use_prior_only,
            config=Config(
                dataPath=str(DATA),
                exclude=None,
                registration=registration,
            ),
            overwrite=overwrite,
            repaired_rounds=repaired_rounds,
            max_iters=max_iters,
            use_shifts_from=use_shifts_from,
        )


@click.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.argument("roi", type=str, default="*")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path, resolve_path=True),
    default=None,
    help=(
        "Path to a JSON RegisterConfig (or Config containing 'registration'). "
        "Defaults to <workspace>/analysis/deconv/config.json then <workspace>/config.json when present. "
        "Batch forwards it to child runs."
    ),
)
@click.option(
    "--codebook",
    help="Path to the codebook file",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
)
@click.option("--reference", "--ref", default=None, help="Reference identifier")
@click.option("--fwhm", type=float, default=None, show_default="from config", help="FWHM value")
@click.option("--threshold", type=float, default=None, show_default="from config", help="Threshold value")
@click.option("--threads", type=int, default=15, help="Number of threads to use")
@click.option("--overwrite", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option(
    "--only-median-gt",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help=(
        "Only process tiles whose per-round L2 distance from the median shift exceeds this value, "
        "as recorded by `preprocess check-shifts` (output/shifts_metrics). Requires --overwrite."
    ),
)
@click.option(
    "--only-corr-lt",
    type=click.FloatRange(min=-1.0, max=1.0),
    default=None,
    help=(
        "Only process tiles whose per-round correlation is below this value, "
        "as recorded by `preprocess check-shifts` (output/shifts_metrics). Requires --overwrite."
    ),
)
@click.option(
    "--verify",
    is_flag=True,
    help=(
        "After batch registration, verify each registered TIFF can be read and has a "
        "consistent shape (based on the first registered file). If a file fails, rerun "
        "the single 'run' command with --overwrite."
    ),
)
@click.option(
    "--use-fft",
    is_flag=True,
    default=None,
    help="Use FFT phase correlation instead of spot-based matching.",
)
@click.option(
    "--use-itk",
    is_flag=True,
    default=None,
    help="Use SimpleITK gradient descent for alignment (more robust for low-contrast).",
)
@click.option(
    "--use-brightest",
    type=int,
    default=None,
    show_default="from config",
    help="If >0, use only the N brightest fiducial spots per image for spot-based alignment.",
)
@click.option(
    "--offset-brightest",
    type=click.IntRange(min=0),
    default=None,
    show_default="from config",
    help="Skip the first N brightest fiducial spots before applying --use-brightest.",
)
@click.option(
    "--allow-large-shifts",
    is_flag=True,
    default=None,
    help="Accept shifts larger than the configured threshold instead of raising DriftTooLarge.",
)
@click.option(
    "--repaired",
    type=str,
    default=None,
    help="Comma-separated round names to use from --repaired folders (e.g., '1_9_17,2_10_18')",
)
@click.option(
    "--max-iters",
    type=int,
    default=5,
    show_default=True,
    help="Maximum iterations for spot-based drift refinement.",
)
@click.option(
    "--use-shifts-from",
    type=str,
    default=None,
    help="Codebook name to copy shifts from, skipping fiducial registration entirely.",
)
@click.option(
    "--use-prior-only",
    is_flag=True,
    help="Skip automatic registration and use only the prior shifts (after shifting fiducials, residual shift is 0,0).",
)
def batch(
    path: Path,
    roi: str,
    config_path: Path | None,
    reference: str | None,
    codebook: Path,
    fwhm: float | None,
    threshold: float | None,
    threads: int,
    overwrite: bool,
    debug: bool,
    only_median_gt: float | None,
    only_corr_lt: float | None,
    verify: bool,
    use_fft: bool | None = None,
    use_itk: bool | None = None,
    use_brightest: int | None = None,
    offset_brightest: int | None = None,
    allow_large_shifts: bool | None = None,
    repaired: str | None = None,
    max_iters: int = 5,
    use_shifts_from: str | None = None,
    use_prior_only: bool = False,
):
    ws = Workspace(path)
    if config_path is None:
        config_path = ws.config_json()
        if config_path is None:
            raise click.ClickException(
                "No config.json found. Provide --config or create "
                "'analysis/deconv/config.json' (preferred) or 'config.json' at the workspace root."
            )
    # idxs = None
    # use_custom_idx = idxs is not None
    codebook = _copy_codebook_to_workspace(path, codebook)
    codebook_name = codebook.stem
    setup_cli_logging(
        path,
        component="preprocess.register.batch",
        file=f"register-batch-{codebook_name}",
        debug=debug,
        extra={"codebook": codebook_name},
    )
    _silence_matplotlib_debug_logs()
    ws = Workspace(path)
    logger.info(f"Found {ws.rois}")

    if only_median_gt is not None and not overwrite:
        raise click.ClickException("--only-median-gt requires --overwrite.")
    if only_corr_lt is not None and not overwrite:
        raise click.ClickException("--only-corr-lt requires --overwrite.")
    if use_prior_only and use_shifts_from is not None:
        raise click.ClickException("--use-prior-only cannot be combined with --use-shifts-from.")

    loaded_registration = _load_register_config_from_json(config_path)
    _copy_chromatic_corrections_to_output(path, loaded_registration.chromatic_path)

    registration_effective = _apply_cli_overrides_to_registration_config(
        loaded_registration,
        reference=reference,
        threshold=threshold,
        fwhm=fwhm,
        use_fft=use_fft,
        use_itk=use_itk,
        use_brightest=use_brightest,
        offset_brightest=offset_brightest,
        allow_large_shifts=allow_large_shifts,
    )
    reference_effective = registration_effective.reference

    if (
        registration_effective.fiducial.detailed.offset_brightest > 0
        and registration_effective.fiducial.detailed.use_brightest <= 0
    ):
        raise click.ClickException("--offset-brightest requires --use-brightest > 0.")

    # Progressive scoping: default to all ROIs ("*"/"all"), otherwise process the
    # single provided ROI to align with other preprocess CLIs.
    selected_rois = ws.rois if roi in {"*", "all"} else [roi]
    for roi in selected_rois:
        if use_shifts_from:
            shift_dir = ws.shifts(roi, use_shifts_from)
            shift_files = sorted(shift_dir.glob("shifts-*.json"))
            if not shift_files:
                raise ValueError(f"No shift files found for {roi}+{use_shifts_from} in {shift_dir}")
            all_idxs = sorted({int(p.stem.split("-")[1]) for p in shift_files})
        else:
            names = sorted(
                {name for name in path.rglob(f"{reference_effective}--{roi}/{reference_effective}*.tif")}
            )
            if not len(names):
                fid_dir = ws.fids(roi)
                fid_files = sorted(fid_dir.glob("fids-*.tif")) if fid_dir.exists() else []
                fid_idxs: list[int] = []
                for p in fid_files:
                    try:
                        fid_idxs.append(int(p.stem.rsplit("-", 1)[-1]))
                    except ValueError:
                        continue

                if not fid_idxs:
                    raise ValueError(f"No images found for {reference_effective}--{roi}")

                logger.warning(
                    f"No images found for {reference_effective}--{roi}; using fiducials in {fid_dir} to determine indices."
                )
                all_idxs = sorted(set(fid_idxs))
            else:
                all_idxs = sorted({int(name.stem.split("-")[1]) for name in names})

        wants_metrics_filter = only_median_gt is not None or only_corr_lt is not None
        if wants_metrics_filter:
            metrics_output_dir = ws.output.root
            metrics_output_dir.mkdir(parents=True, exist_ok=True)
            _run_child_cli(
                [
                    "preprocess",
                    "check-shifts",
                    str(ws.path),
                    str(roi),
                    "--codebook",
                    str(codebook),
                    "--output",
                    str(metrics_output_dir),
                ],
                check=True,
            )

            metrics_csv = metrics_output_dir / "shifts_metrics" / f"shifts_metrics--{roi}+{codebook_name}.csv"
            selected: set[int] = set()
            if only_median_gt is not None:
                selected.update(
                    _load_outlier_tiles_from_shifts_metrics(metrics_csv, only_median_gt=only_median_gt)
                )
            if only_corr_lt is not None:
                selected.update(_load_low_corr_tiles_from_shifts_metrics(metrics_csv, only_corr_lt=only_corr_lt))

            all_idxs = [i for i in all_idxs if i in selected]
            if not all_idxs and not verify:
                parts: list[str] = []
                if only_median_gt is not None:
                    parts.append(f"L2 > {only_median_gt}")
                if only_corr_lt is not None:
                    parts.append(f"corr < {only_corr_lt}")
                filters = " or ".join(parts) if parts else "the requested filter"
                logger.warning(
                    f"Skipping {reference_effective}--{roi}: no tiles found with {filters}."
                )
                continue

        idxs = [i for i in all_idxs if overwrite or not ws.regimg(roi, codebook.stem, i).exists()]

        if not idxs and not verify:
            logger.warning(f"Skipping {reference_effective}--{roi}, already registered.")
            continue

        with progress_bar_threadpool(len(idxs), threads=threads, debug=debug) as submit:
            for i in idxs:
                submit(
                    _run_child_cli,
                    _build_register_run_argv(
                        path=path,
                        idx=i,
                        codebook=codebook,
                        config=config_path,
                        fwhm=fwhm,
                        threshold=threshold,
                        reference=reference,
                        roi=roi,
                        max_iters=max_iters,
                        overwrite=overwrite,
                        debug=debug,
                        repaired=repaired,
                        use_fft=use_fft,
                        use_itk=use_itk,
                        use_brightest=use_brightest,
                        offset_brightest=offset_brightest,
                        allow_large_shifts=allow_large_shifts,
                        use_shifts_from=use_shifts_from,
                        use_prior_only=use_prior_only,
                    ),
                    check=True,
                )

        # Optional verification phase: ensure files are readable and shapes match.
        if verify:
            codebook_name = codebook.stem
            reg_dir = ws.registered(roi, codebook_name)
            expected_shape: tuple[int, int, int, int] | None = None
            verify_idxs = all_idxs

            def _read_shape(p: Path) -> tuple[int, int, int, int]:
                try:
                    with TiffFile(p) as tif:  # Verify decoding and read shape
                        arr = tif.asarray()
                        shape = tuple(arr.shape)
                        assert len(shape) == 4, f"Unexpected ndim {arr.ndim} for {p}"
                        return shape  # type: ignore[return-value]
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"Failed to read registered file {p}: {e}") from e

            # Establish baseline expected shape from the first index that exists and is readable
            for i in verify_idxs:
                p = reg_dir / f"reg-{i:04d}.tif"
                if not p.exists():
                    # If the file does not exist (e.g., child run skipped), skip baseline attempt.
                    continue
                try:
                    expected_shape = _read_shape(p)
                    logger.info(f"[{roi}] Baseline registered shape from {p.name}: {expected_shape}")
                    break
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[{roi}] Baseline read failed for {p.name}: {e}")

            # Verify each produced file; if any fails, rerun the single index with overwrite
            failed: list[int] = []
            for i in verify_idxs:
                p = reg_dir / f"reg-{i:04d}.tif"
                if not p.exists():
                    logger.warning(f"[{roi}] Missing output {p.name}; scheduling rerun.")
                    failed.append(i)
                    continue
                try:
                    shape = _read_shape(p)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[{roi}] Read failed for {p.name}: {e}; scheduling rerun.")
                    failed.append(i)
                    continue

                if expected_shape is not None and shape != expected_shape:
                    logger.warning(
                        f"[{roi}] Shape mismatch for {p.name}: got {shape}, expected {expected_shape}; scheduling rerun."
                    )
                    failed.append(i)

            for i in failed:
                logger.info(f"[{roi}] Re-running index {i:04d} with --overwrite due to verification failure.")
                _run_child_cli(
                    _build_register_run_argv(
                        path=path,
                        idx=i,
                        codebook=codebook,
                        config=config_path,
                        fwhm=fwhm,
                        threshold=threshold,
                        reference=reference,
                        roi=roi,
                        max_iters=max_iters,
                        overwrite=True,
                        debug=debug,
                        repaired=repaired,
                        use_fft=use_fft,
                        use_itk=use_itk,
                        use_brightest=use_brightest,
                        offset_brightest=offset_brightest,
                        allow_large_shifts=allow_large_shifts,
                        use_shifts_from=use_shifts_from,
                        use_prior_only=use_prior_only,
                    ),
                    check=True,
                )

                # Re-verify just this file; log if still failing
                p = reg_dir / f"reg-{i:04d}.tif"
                try:
                    shape = _read_shape(p)
                    if expected_shape is not None and shape != expected_shape:
                        logger.error(
                            f"[{roi}] Post-rerun shape still mismatched for {p.name}: {shape} vs {expected_shape}."
                        )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"[{roi}] Post-rerun read still failing for {p.name}: {e}")

        if wants_metrics_filter:
            # Refresh diagnostics after potentially re-registering outlier tiles.
            metrics_output_dir = ws.output.root
            metrics_output_dir.mkdir(parents=True, exist_ok=True)
            _run_child_cli(
                [
                    "preprocess",
                    "check-shifts",
                    str(ws.path),
                    str(roi),
                    "--codebook",
                    str(codebook),
                    "--output",
                    str(metrics_output_dir),
                ],
                check=True,
            )


register.add_command(batch)


def _load_shifts_from_codebook(
    ws: Workspace,
    *,
    roi: str,
    source_codebook: str,
    idx: int,
) -> dict[str, np.ndarray]:
    """Load shifts from another codebook's shift files.

    Args:
        ws: Workspace instance.
        roi: ROI name.
        source_codebook: Codebook name to load shifts from.
        idx: Tile index.

    Returns:
        Dictionary mapping round names to shift vectors [dx, dy].

    Raises:
        FileNotFoundError: If the shift file doesn't exist for the source codebook.
    """
    shift_path = ws.shift_json(roi, source_codebook, idx)
    if not shift_path.exists():
        raise FileNotFoundError(
            f"Shift file not found for codebook '{source_codebook}' at {shift_path}. "
            f"Run registration with that codebook first."
        )

    shift_data = Shifts.validate_json(shift_path.read_text())
    return {name: np.array(s.shifts) for name, s in shift_data.items()}


def _load_reference_fid_from_previous_run(
    ws: Workspace,
    *,
    roi: str,
    reference: str,
    idx: int,
    prefer_codebook: str | None = None,
) -> np.ndarray:
    """Load a reference fiducial plane when the reference round TIFF is missing.

    Preference order:
    1) Any previously written registered fiducial stacks (raw fiducials) under
       ``registered--ROI+codebook/_fids-XXXX.tif``.
    2) The ROI-scoped fiducial thumbnail under ``fids--ROI/fids-XXXX.tif`` when present.
    """

    codebooks = ws.registered_codebooks(rois=[roi])
    ordered: list[str] = []
    if prefer_codebook is not None:
        ordered.append(prefer_codebook)
    ordered.extend([cb for cb in codebooks if cb != prefer_codebook])

    for codebook in ordered:
        fids_path = ws.registered_fids(roi, codebook) / f"_fids-{idx:04d}.tif"
        if not fids_path.exists():
            continue

        with TiffFile(fids_path) as tif:
            stack = tif.asarray()
            metadata: dict[str, Any] | None = None
            try:
                metadata = tif.shaped_metadata[0]  # type: ignore[index]
            except (AttributeError, IndexError, TypeError):
                metadata = tif.imagej_metadata

            keys: list[str] | None = None
            if isinstance(metadata, dict):
                raw_keys = metadata.get("key")
                if isinstance(raw_keys, list) and all(isinstance(k, str) for k in raw_keys):
                    keys = raw_keys
                elif isinstance(raw_keys, str):
                    try:
                        decoded = json.loads(raw_keys)
                    except json.JSONDecodeError:
                        decoded = None
                    if isinstance(decoded, list) and all(isinstance(k, str) for k in decoded):
                        keys = decoded

            if keys is None:
                continue

            try:
                plane_idx = keys.index(reference)
            except ValueError:
                continue

            plane = np.asarray(stack[plane_idx]).astype(np.float32)
            logger.info(f"Loaded reference fiducial from previous run: {fids_path}")
            return plane

    fid_path = ws.fid(roi, idx)
    if fid_path.exists():
        with TiffFile(fid_path) as tif:
            plane = np.asarray(tif.asarray()).astype(np.float32)
        logger.info(f"Loaded reference fiducial from {fid_path} (fids directory fallback)")
        return plane

    searched = ", ".join(ordered) if ordered else "none"
    raise FileNotFoundError(
        f"Reference round directory is missing and no previous-run fiducial was found for "
        f"reference={reference} idx={idx:04d} roi={roi}. Searched codebooks: {searched}."
    )


@register.command("fix-shifts")
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path, resolve_path=True))
@click.argument("idx", required=False, type=int)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path, resolve_path=True),
    default=None,
    help=(
        "Path to a JSON RegisterConfig (or Config containing 'registration'). "
        "Defaults to <workspace>/analysis/deconv/config.json then <workspace>/config.json when present."
    ),
)
@click.option(
    "--roi",
    "-o",
    "roi_option",
    type=str,
    required=False,
    help="ROI to process. Use '*' for all ROIs.",
)
@click.option("--reference", "-r", type=str, default=None, show_default="from config", help="Reference round name")
@click.option(
    "--rounds",
    type=str,
    required=True,
    help="Comma-separated list of rounds to fix (e.g., '1_9_17,3_11_19')",
)
@click.option("--use-fft/--use-spots", default=None, show_default="from config", help="Use FFT phase correlation or spot-based matching.")
@click.option("--n-fids", type=int, default=None, show_default="from config", help="Number of fiducial frames per image")
@click.option("--threshold", type=float, default=None, show_default="from config", help="Spot detection threshold (sigma)")
@click.option("--fwhm", type=float, default=None, show_default="from config", help="Fiducial spot FWHM")
@click.option(
    "--use-brightest",
    type=int,
    default=None,
    show_default="from config",
    help="If >0, use only the N brightest fiducials (matches register run).",
)
@click.option(
    "--offset-brightest",
    type=click.IntRange(min=0),
    default=None,
    show_default="from config",
    help="Skip the first N brightest fiducial spots before applying --use-brightest.",
)
@click.option(
    "--allow-large-shifts/--strict-shifts",
    default=None,
    show_default="from config",
    help="Enable the same allow_large_shifts toggle used in register run.",
)
@click.option(
    "--priors",
    type=str,
    default=None,
    help="Prior shift as 'dx,dy' for the target round, or path to JSON/CSV mapping round→dx,dy.",
)
@click.option("--debug", is_flag=True)
def fix_shifts(
    path: Path,
    idx: int | None,
    config_path: Path | None,
    roi_option: str | None,
    reference: str | None,
    rounds: str,
    use_fft: bool | None,
    n_fids: int | None,
    threshold: float | None,
    fwhm: float | None,
    use_brightest: int | None,
    offset_brightest: int | None,
    allow_large_shifts: bool | None,
    priors: str | None,
    debug: bool,
):
    """Detect large drift offsets and write shifts to JSON.

    Uses the SAME registration algorithm as the main pipeline (align_fiducials)
    but only detects and writes shifts - does NOT apply them to images.

    The output JSON can be used to inform priors for `register batch`.

    If IDX is provided, only that tile is processed.

    Example:

    \b
        # Detect large shifts
        preprocess register fix-shifts /ws 1 --roi roi1 \\
            --rounds "1_9_17,3_11_19" --reference 2_10_18

    \b
        # Review detected shifts
        cat /ws/analysis/deconv/shifts--roi1/coarse_shifts.json
    """
    import json as json_module

    ws = Workspace(path)
    if config_path is None:
        config_path = ws.config_json()
        if config_path is None:
            raise click.ClickException(
                "No config.json found. Provide --config or create "
                "'analysis/deconv/config.json' (preferred) or 'config.json' at the workspace root."
            )

    loaded_registration = _load_register_config_from_json(config_path)
    _copy_chromatic_corrections_to_output(path, loaded_registration.chromatic_path)
    staged_chromatic_dir = ws.output.chromatic
    roi_value = roi_option
    if roi_value is None:
        raise click.ClickException("ROI is required. Pass it with --roi.")
    if roi_value == "*":
        rois = ws.resolve_rois()
    else:
        rois = ws.resolve_rois([roi_value])
    rounds_to_fix = [r.strip() for r in rounds.split(",")]

    cli_priors: dict[str, tuple[float, float]] | None = None
    if priors is not None:
        priors_value = priors.strip()
        parts = [p.strip() for p in priors_value.split(",")]
        if len(parts) == 2:
            if len(rounds_to_fix) != 1:
                raise click.ClickException(
                    "--priors as 'dx,dy' requires exactly one --rounds entry."
                )
            dx, dy = float(parts[0]), float(parts[1])
            cli_priors = {rounds_to_fix[0]: (dx, dy)}
        else:
            cli_priors = _parse_priors_file(Path(priors_value))

    if cli_priors:
        logger.info(f"Using explicit priors for rounds: {sorted(cli_priors)}")

    reference_effective = reference or loaded_registration.reference
    priors_update = (cli_priors or {}) if priors is not None else None
    registration = _apply_cli_overrides_to_registration_config(
        loaded_registration,
        reference=reference_effective,
        threshold=threshold,
        fwhm=fwhm,
        use_fft=use_fft,
        n_fids=n_fids,
        use_brightest=use_brightest,
        offset_brightest=offset_brightest,
        allow_large_shifts=allow_large_shifts,
        priors=priors_update,
    ).model_copy(update={"chromatic_path": staged_chromatic_dir})

    if (
        registration.fiducial.detailed.offset_brightest > 0
        and registration.fiducial.detailed.use_brightest <= 0
    ):
        raise click.ClickException("--offset-brightest requires --use-brightest > 0.")

    config = Config(
        dataPath=str(DATA),
        exclude=None,
        registration=registration,
    )
    fiducial_cfg = config.registration.fiducial

    for roi in rois:
        setup_cli_logging(
            path,
            component="preprocess.register.fix-shifts",
            file=f"fix-shifts-{roi}",
            debug=debug,
            extra={"roi": roi, "reference": reference_effective},
        )
        _silence_matplotlib_debug_logs()

        logger.info(f"Processing rounds: {rounds_to_fix}")
        logger.info(f"Reference round: {reference_effective}")
        logger.info(f"ROI: {roi}")

        ref_dir = path / f"{reference_effective}--{roi}"
        if not ref_dir.exists():
            raise ValueError(f"Reference round directory not found: {ref_dir}")

        for round_name in rounds_to_fix:
            round_dir = path / f"{round_name}--{roi}"
            if not round_dir.exists():
                raise ValueError(f"Round directory not found: {round_dir}")

        if idx is not None:
            ref_path = ref_dir / f"{reference_effective}-{idx:04d}.tif"
            if not ref_path.exists():
                raise ValueError(f"Reference tile not found: {ref_path}")
            for round_name in rounds_to_fix:
                round_path = path / f"{round_name}--{roi}" / f"{round_name}-{idx:04d}.tif"
                if not round_path.exists():
                    raise ValueError(f"Round tile not found: {round_path}")
            idxs = [idx]
        else:
            ref_tiles = sorted(ref_dir.glob(f"{reference_effective}-*.tif"))
            if not ref_tiles:
                raise ValueError(f"No tiles found in reference directory: {ref_dir}")
            idxs = [int(p.stem.split("-")[1]) for p in ref_tiles]
        logger.info(f"Found {len(idxs)} tiles to process")

        all_results: dict[int, dict[str, dict[str, float]]] = {}

        for idx in idxs:
            logger.info(f"Processing tile {idx}")

            fids: dict[str, np.ndarray] = {}
            use_raw = fiducial_cfg.use_fft or fiducial_cfg.use_itk

            ref_path = ref_dir / f"{reference_effective}-{idx:04d}.tif"
            ref_img = Image.from_file(ref_path, n_fids=fiducial_cfg.n_fids)
            fids[reference_effective] = ref_img.fid_raw.astype(np.float32) if use_raw else ref_img.fid

            fid_raw_images: dict[str, np.ndarray] | None = None
            if debug:
                fid_raw_images = {reference_effective: ref_img.fid_raw.astype(np.float32)}

            for round_name in rounds_to_fix:
                round_dir = path / f"{round_name}--{roi}"
                target_path = round_dir / f"{round_name}-{idx:04d}.tif"
                if not target_path.exists():
                    logger.warning(f"Target tile not found: {target_path}, skipping")
                    continue
                target_img = Image.from_file(target_path, n_fids=fiducial_cfg.n_fids)
                fid = target_img.fid_raw.astype(np.float32) if use_raw else target_img.fid
                fids[round_name] = fid
                if fid_raw_images is not None:
                    fid_raw_images[round_name] = target_img.fid_raw.astype(np.float32)

            prior_mapping = _apply_priors_to_fids(
                fids,
                fid_raw_images,
                priors=fiducial_cfg.priors,
                anchor_roi=fiducial_cfg.anchor_roi,
                idx=idx,
            )

            shifts, residuals, _ = align_fiducials_with_stats(
                fids,
                reference=reference_effective,
                debug=debug,
                max_iters=5,
                threshold_sigma=fiducial_cfg.threshold,
                fwhm=fiducial_cfg.fwhm,
                use_fft=fiducial_cfg.use_fft,
                use_itk=fiducial_cfg.use_itk,
                use_brightest=fiducial_cfg.detailed.use_brightest,
                detailed_config=fiducial_cfg.detailed,
            )

            _add_priors_to_shifts(
                shifts,
                priors=fiducial_cfg.priors,
                prior_mapping=prior_mapping,
                anchor_roi=fiducial_cfg.anchor_roi,
            )

            tile_results: dict[str, dict[str, float]] = {}
            for round_name in rounds_to_fix:
                if round_name in shifts:
                    dx = float(shifts[round_name][0])
                    dy = float(shifts[round_name][1])
                    magnitude = float(np.hypot(dx, dy))
                    tile_results[round_name] = {
                        "dx": dx,
                        "dy": dy,
                        "magnitude": magnitude,
                        "residual": float(residuals.get(round_name, 0.0)),
                    }
                    logger.info(f"  {round_name}: dx={dx:.2f}, dy={dy:.2f}, mag={magnitude:.1f}")

            all_results[idx] = tile_results

        shifts_dir = ws.shifts(roi)
        shifts_dir.mkdir(exist_ok=True)
        output_path = ws.coarse_shifts_json(roi)

        existing_tiles: dict[str, Any] = {}
        existing_priors: dict[str, Any] | None = None
        output_reference = reference_effective
        output_use_fft = fiducial_cfg.use_fft
        if output_path.exists():
            try:
                existing_data = json_module.loads(output_path.read_text())
            except json_module.JSONDecodeError as exc:
                raise click.ClickException(
                    f"Existing coarse shifts JSON is invalid: {output_path}"
                ) from exc
            if not isinstance(existing_data, dict):
                raise click.ClickException(
                    f"Existing coarse shifts JSON must be an object: {output_path}"
                )
            if "reference" in existing_data and existing_data["reference"] != reference_effective:
                logger.warning(
                    "Existing coarse shifts reference "
                    f"'{existing_data['reference']}' does not match '{reference_effective}'. "
                    "Keeping the existing reference to avoid clobbering metadata."
                )
                output_reference = existing_data["reference"]
            if "use_fft" in existing_data and existing_data["use_fft"] != fiducial_cfg.use_fft:
                logger.warning(
                    "Existing coarse shifts use_fft "
                    f"{existing_data['use_fft']} does not match {fiducial_cfg.use_fft}. "
                    "Keeping the existing use_fft to avoid clobbering metadata."
                )
                output_use_fft = existing_data["use_fft"]
            tiles_value = existing_data.get("tiles")
            if isinstance(tiles_value, dict):
                existing_tiles = tiles_value
            existing_priors_value = existing_data.get("priors")
            if isinstance(existing_priors_value, dict):
                existing_priors = existing_priors_value

        updated_tiles = {f"{idx:04d}": results for idx, results in sorted(all_results.items())}
        merged_tiles = {**existing_tiles, **updated_tiles}

        output_data: dict[str, Any] = {
            "reference": output_reference,
            "use_fft": output_use_fft,
            "tiles": merged_tiles,
        }
        if cli_priors is not None:
            output_data["priors"] = {
                name: {"dx": dx, "dy": dy}
                for name, (dx, dy) in sorted(cli_priors.items())
            }
        elif existing_priors is not None:
            output_data["priors"] = existing_priors
        output_path.write_text(json_module.dumps(output_data, indent=2))
        logger.info(f"Wrote coarse shifts to {output_path}")

        for round_name in rounds_to_fix:
            per_round = [r[round_name] for r in all_results.values() if round_name in r]
            magnitudes = [rec["magnitude"] for rec in per_round]
            if magnitudes:
                dx_vals = [rec["dx"] for rec in per_round]
                dy_vals = [rec["dy"] for rec in per_round]
                median_dx = np.median(dx_vals)
                median_dy = np.median(dy_vals)
                logger.info(
                    f"{round_name}: median=({median_dx:.1f}, {median_dy:.1f}), "
                    f"mean mag={np.mean(magnitudes):.1f}px, max mag={np.max(magnitudes):.1f}px"
                )
                max_abs_dx = float(np.max(np.abs(dx_vals)))
                max_abs_dy = float(np.max(np.abs(dy_vals)))
                if max_abs_dx < 35.0 and max_abs_dy < 35.0:
                    logger.warning(
                        f"{round_name}: all detected drifts are <35 px in both X and Y; "
                        "fix-shifts may not be necessary for this round."
                    )


if __name__ == "__main__":
    register()

# %%
