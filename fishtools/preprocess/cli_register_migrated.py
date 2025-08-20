# %%
import json
import pickle
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import tifffile
import toml
import typer
from basicpy import BaSiC
from loguru import logger
from pydantic import ValidationError
from rich.console import Console
from scipy import ndimage
from scipy.ndimage import shift
from tifffile import TiffFile

from fishtools.preprocess.chromatic import Affine
from fishtools.preprocess.cli_deconv import scale_deconv
from fishtools.preprocess.config import (
    Config,
    NumpyEncoder,
)
from fishtools.preprocess.config_loader import (
    load_config_from_toml,
    load_minimal_config,
)
from fishtools.preprocess.fiducial import Shifts, align_fiducials
from fishtools.preprocess.processing_context import ProcessingContext
from fishtools.utils.pretty_print import progress_bar_threadpool
from fishtools.utils.utils import initialize_logger

# FORBIDDEN_PREFIXES moved to config.system.forbidden_prefixes - see ProcessingContext


def spillover_correction(spillee: np.ndarray, spiller: np.ndarray, corr: float):
    scaled = spiller * corr
    return np.where(spillee >= scaled, spillee - scaled, 0)


def parse_nofids(
    nofids: dict[str, np.ndarray],
    shifts: dict[str, np.ndarray],
    channels: dict[str, str],
    context: ProcessingContext,
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

        # Apply configurable spillover correction
        spillover_corrections = context.img_config.spillover_corrections
        if spillover_corrections:
            # Create channel-to-bit mapping
            cs = {str(channels[bit]): bit for bit in curr_bits}

            # Apply spillover corrections based on configuration
            for correction_name, factor in spillover_corrections.items():
                # Parse correction name (e.g., "560_647" -> ["560", "647"])
                if "_" in correction_name:
                    source_channel, target_channel = correction_name.split("_", 1)
                    if source_channel in cs and target_channel in cs:
                        logger.debug(
                            f"Applying spillover correction: {source_channel} -> {target_channel} (factor: {factor})"
                        )
                        out[cs[source_channel]] = spillover_correction(
                            out[cs[source_channel]], out[cs[target_channel]], factor
                        )

    for name, shift_ in shifts.items():
        curr_bits = name.split("-")[0].split("_")
        for i, bit in enumerate(curr_bits):
            out_shift[bit] = shift_

    return out, out_shift, bit_name_mapping


def sort_key(x: tuple[str, np.ndarray]) -> int | str:
    """Key function for dict.items() by numerical order.

    Args:
        x: dict.items() tuple with bit name as first element and array as second element.

    Returns:
        Key for sorting.
    """
    try:
        return f"{int(x[0]):02d}"
    except ValueError:
        return x[0]


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
    global_deconv_scaling: np.ndarray
    basic: Callable[[], dict[str, BaSiC] | None]

    # CHANNELS moved to config.system.available_channels - loaded dynamically from context

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        context: ProcessingContext,
        discards: dict[str, list[str]] | None = None,
        n_fids: int = 2,
    ):
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
                img = tif.asarray()
                try:
                    metadata = tif.shaped_metadata[0]  # type: ignore
                except IndexError:
                    metadata = tif.imagej_metadata
            except (
                IndexError
            ) as e:  # tifffile throws IndexError if the file is truncated
                raise Exception(
                    f"File {path} is corrupted. Please check the file."
                ) from e
        try:
            assert metadata is not None
            waveform = (
                metadata["waveform"]
                if isinstance(metadata["waveform"], dict)
                else json.loads(metadata["waveform"])
            )
        except KeyError:
            waveform = toml.load(path.with_name(f"{path.name.split('-')[0]}.toml"))

        # Use configured available channels
        channels = [f"ilm{n}" for n in context.system_config.available_channels]
        counts = {key: sum(waveform[key]["sequence"]) for key in channels}

        # To remove ilm from, say, ilm405.

        powers = {
            key[3:]: waveform[key]["power"]
            for key in channels
            if (key == "ilm405" and counts[key] > n_fids)
            or (key != "ilm405" and counts[key])
        }

        if waveform.get("params"):
            powers = waveform["params"]["powers"]

        if len(powers) != len(bits):
            raise ValueError(
                f"{path}: Expected {len(bits)} channels, got {len(powers)}"
            )

        try:
            global_deconv_scaling = (
                np.loadtxt(path.parent.parent / "deconv_scaling" / f"{name}.txt")
                .astype(np.float32)
                .reshape((2, -1))
            )
        except FileNotFoundError:
            if "deconv" in path.resolve().as_posix():
                raise ValueError("No deconv_scaling found.")
            logger.debug("No deconv_scaling found. Using ones.")
            global_deconv_scaling = np.ones((2, len(bits)))

        nofid = img[:-n_fids].reshape(
            -1,
            len(powers),
            context.img_config.image_size,
            context.img_config.image_size,
        )

        if to_discard_idxs:
            _bits = name.split("_")
            power_keys = list(powers.keys())
            for _idx_d in to_discard_idxs:
                _bits.pop(_idx_d)
                del powers[power_keys[_idx_d]]
            name = "_".join(_bits)
            keeps = list(sorted(set(range(len(bits))) - set(to_discard_idxs)))
            nofid = nofid[:, keeps]
            bits = [bits[i] for i in keeps]
            global_deconv_scaling = global_deconv_scaling[:, keeps]
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

        fids_raw = np.atleast_3d(img[-n_fids:]).max(axis=0)
        return cls(
            name=name,
            idx=int(idx),
            nofid=nofid,
            fid=cls.loG_fids(fids_raw, context),
            fid_raw=fids_raw,
            bits=bits,
            powers=powers,
            metadata=metadata,
            global_deconv_scaling=global_deconv_scaling,
            basic=basic,
        )

    @staticmethod
    def loG_fids(fid: np.ndarray, context: ProcessingContext):
        """Apply Laplacian of Gaussian filter with configurable parameters.

        Args:
            fid: Input fiducial image data
            context: Processing context with configuration parameters

        Returns:
            Normalized LoG-filtered image
        """
        # Fix Bug 1: Apply max projection for 3D input, then process the 2D result
        if len(fid.shape) == 3:
            fid = fid.max(axis=0)

        # Use configurable sigma parameter
        temp = -ndimage.gaussian_laplace(
            fid.astype(np.float32).copy(), sigma=context.img_config.log_sigma
        )
        temp -= temp.min()

        # Use configurable percentiles
        percs = np.percentile(temp, context.img_config.percentiles)

        # Fix Bug 2: Handle uniform data to prevent division by zero
        if percs[1] - percs[0] == 0:
            # For uniform data, return array of zeros (normalized)
            temp = np.zeros_like(temp)
        else:
            temp = (temp - percs[0]) / (percs[1] - percs[0])

        return temp


def run_fiducial(
    path: Path,
    fids: dict[str, np.ndarray],
    codebook_name: str,
    config: Config,
    context: ProcessingContext,
    *,
    roi: str,
    idx: int,
    reference: str,
    debug: bool,
    no_priors: bool = False,
):
    logger.info("Aligning fiducials")

    prior_mapping: dict[str, str] = {}

    if (
        not config.registration.fiducial.use_fft
        and len(
            shifts_existing := sorted(
                (path / f"shifts--{roi}+{codebook_name}").glob("*.json")
            )
        )
        > context.img_config.priors_min_count
        and not no_priors
    ):
        _priors: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for shift_path in shifts_existing:
            try:
                shift_dicts = Shifts.validate_json(shift_path.read_text())
            except ValidationError as e:
                logger.warning(f"Error decoding {shift_path} {e}. Skipping.")
                continue
            for name, shift_dict in shift_dicts.items():
                if shift_dict.residual < context.img_config.residual_threshold:
                    _priors[name].append(shift_dict.shifts)
        logger.debug(f"Using priors from {len(shifts_existing)} existing shifts.")
        config.registration.fiducial.priors = {
            name: tuple(np.median(np.array(shifts), axis=0))
            for name, shifts in _priors.items()
        }
        logger.debug(config.registration.fiducial.priors)

    if config.registration.fiducial.priors is not None:
        for name, sh in config.registration.fiducial.priors.items():
            for file in fids:
                if file.startswith(name):
                    fids[file] = shift(fids[file], [sh[1], sh[0]], order=1)
                    prior_mapping[name] = file
                    break
            else:
                raise ValueError(
                    f"{idx}: Searched {list(fids.keys())}. Could not find file that starts with {name} for prior shift."
                )

    if config.registration.fiducial.overrides is not None:
        for name, sh in config.registration.fiducial.overrides.items():
            for file in fids:
                if file.startswith(name):
                    logger.info(f"Overriding shift for {name} to {sh}")
                    fids[file] = shift(fids[file], [sh[1], sh[0]], order=1)
                    prior_mapping[name] = file
                    break
            else:
                raise ValueError(
                    f"Could not find file that starts with {name} for override shift."
                )

    # Write reference fiducial
    (fid_path := path / f"fids--{roi}").mkdir(exist_ok=True)
    crop, downsample = config.registration.crop, config.registration.downsample
    # Use configured compression settings
    compression_level = context.stitching_config.compression_levels.get("medium", 0.65)
    tifffile.imwrite(
        fid_path / f"fids-{idx:04d}.tif",
        fids[reference][crop:-crop, crop:-crop],
        compression=22610,
        compressionargs={"level": compression_level},
        metadata={"axes": "YX"},
    )

    _fids_path = path / f"registered--{roi}+{codebook_name}" / "_fids"
    _fids_path.mkdir(exist_ok=True, parents=True)

    # Use configured compression settings
    compression_level = context.stitching_config.compression_levels.get("medium", 0.65)
    tifffile.imwrite(
        _fids_path / f"_fids-{idx:04d}.tif",
        np.stack([v for v in fids.values()]),
        compression=22610,
        compressionargs={"level": compression_level},
        metadata={"axes": "CYX", "key": list(fids.keys())},
    )

    shifts, residuals = align_fiducials(
        fids,
        reference=reference,
        debug=debug,
        max_iters=context.img_config.max_iterations,
        threshold_sigma=config.registration.fiducial.threshold,
        fwhm=config.registration.fiducial.fwhm,
        use_fft=config.registration.fiducial.use_fft,
        detailed_config=config.registration.fiducial.detailed,
    )

    assert shifts  # type: ignore
    assert residuals  # type: ignore

    shifted = {k: shift(fid, [shifts[k][1], shifts[k][0]]) for k, fid in fids.items()}

    if debug:
        # Use configured compression settings
        compression_level = context.stitching_config.compression_levels.get(
            "medium", 0.65
        )
        tifffile.imwrite(
            path / f"fids_shifted-{idx:04d}.tif",
            # Prior shifts already applied.
            np.stack(list(shifted.values())),
            compression=22610,
            compressionargs={"level": compression_level},
            metadata={"axes": "CYX"},
        )

    if config.registration.fiducial.priors is not None:
        for name, sh in config.registration.fiducial.priors.items():
            if (
                not config.registration.fiducial.overrides
                or name not in config.registration.fiducial.overrides
            ):
                shifts[prior_mapping[name]][0] += sh[0]
                shifts[prior_mapping[name]][1] += sh[1]

    (shift_path := path / f"shifts--{roi}+{codebook_name}").mkdir(exist_ok=True)

    start, end, step = (
        context.img_config.correlation_region["start"],
        context.img_config.correlation_region["end"],
        context.img_config.correlation_region["step"],
    )
    _fid_ref = fids[reference][start:end:step, start:end:step].flatten()
    validated = Shifts.validate_python({
        k: {
            "shifts": (shifts[k][0], shifts[k][1]),
            "residual": residuals[k],
            "corr": 1.0
            if reference == k
            else np.corrcoef(
                shifted[k][start:end:step, start:end:step].flatten(), _fid_ref
            )[0, 1],
        }
        for k in fids
    })
    jsoned = Shifts.dump_json(validated)
    (shift_path / f"shifts-{idx:04d}.json").write_bytes(jsoned)
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
):
    logger.info("Reading files")

    # Create processing context with configuration validation
    context = ProcessingContext.create(path, config)
    context.log_processing_parameters()

    out_path = path / f"registered--{roi}+{Path(codebook).stem}"
    out_path.mkdir(exist_ok=True, parents=True)

    if not overwrite and (out_path / f"reg-{idx:04d}.tif").exists():
        logger.info(f"Skipping {idx}")
        return

    shifts = {}

    cb = json.loads(Path(codebook).read_text())
    bits_cb = set(chain.from_iterable(cb.values()))
    folders = {
        p
        for p in Path(path).glob(f"*--{roi}")
        if p.is_dir()
        and not any(
            p.name.startswith(bad)
            for bad in context.system_config.forbidden_prefixes + (config.exclude or [])
        )
        and set(p.name.split("--")[0].split("_"))
        & set(map(str, bits_cb | set(reference.split("_"))))
    }

    # Convert file name to bit
    _imgs: list[Image] = [
        Image.from_file(
            file,
            context=context,
            discards=config.channels and config.channels.discards,
            n_fids=config.registration.fiducial.n_fids,
        )
        for file in chain.from_iterable(p.glob(f"*-{idx:04d}.tif") for p in folders)
        if not any(
            file.parent.name.startswith(bad)
            for bad in context.system_config.forbidden_prefixes + (config.exclude or [])
        )
    ]
    imgs = {img.name: img for img in _imgs}
    del _imgs

    logger.debug(f"{len(imgs)} files: {list(imgs)}")
    if not imgs:
        raise FileNotFoundError(f"No files found in {path} with index {idx}")

    if not shifts:
        shifts = run_fiducial(
            path,
            {name: img.fid for name, img in imgs.items()},
            Path(codebook).stem,
            config,
            context,
            roi=roi,
            reference=reference,
            debug=debug,
            idx=idx,
            no_priors=no_priors,
        )

    for _img in imgs.values():
        del _img.fid, _img.fid_raw

    # Remove reference if not in codebook since we're done with fiducials.
    if not (set(reference.split("_")) & set(map(str, bits_cb))):
        del imgs[reference]

    channels: dict[str, str] = {}
    for img in imgs.values():
        channels |= dict(zip(img.bits, [p[-3:] for p in img.powers]))

    assert all(v.isdigit() for v in channels.values())

    logger.debug(f"Channels: {channels}")

    # Split into individual bits.
    # Spillover correction, max projection
    nofids = {name: img.nofid for name, img in imgs.items()}
    bits, bits_shifted, bit_name_mapping = parse_nofids(
        nofids, shifts, channels, context
    )

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

    transformed: dict[str, np.ndarray] = {}
    ref = None

    # Use context-loaded chromatic correction matrices
    As = {}
    ats = {}
    for channel, (A, t) in context.chromatic_matrices.items():
        As[channel] = A
        ats[channel] = t

    affine = Affine(As=As, ats=ats)
    bits_cb = sorted(set(map(str, bits_cb)) & set(bits))
    for i, bit in enumerate(bits_cb):
        bit = str(bit)
        img = bits[bit]
        del bits[bit]
        c = str(channels[bit])
        # Deconvolution scaling
        orig_name, orig_idx = bit_name_mapping[bit]
        img = collapse_z(img, config.registration.slices).astype(np.float32)
        img = scale_deconv(
            img,
            orig_idx,
            name=orig_name,
            global_deconv_scaling=imgs[orig_name].global_deconv_scaling,
            metadata=imgs[orig_name].metadata,
            debug=debug,
        )

        if ref is None:
            # Need to put this here because of shape change during collapse_z.
            affine.ref_image = ref = img

        # Within-tile alignment. Chromatic corrections.
        logger.debug(f"{bit}: before affine channel={c}, shiftpx={-bits_shifted[bit]}")
        img = affine(img, channel=c, shiftpx=-bits_shifted[bit], debug=debug)
        transformed[bit] = np.clip(img, 0, context.img_config.image_size * 32).astype(
            np.uint16
        )
        logger.debug(f"Transformed {bit}: max={img.max()}, min={img.min()}")

    if not len(transformed):
        raise ValueError("No images were transformed.")

    if not len({img.shape for img in transformed.values()}) == 1:
        raise ValueError(
            f"Transformed images have different shapes: {set(img.shape for img in transformed.values())}"
        )

    crop, downsample = config.registration.crop, config.registration.downsample
    out = np.zeros(
        [
            next(iter(transformed.values())).shape[0],
            # if isinstance(config.registration.slices, slice)
            # else len(config.registration.slices),
            len(transformed),
            (context.img_config.image_size - 2 * crop) // downsample,
            (context.img_config.image_size - 2 * crop) // downsample,
        ],
        dtype=np.uint16,
    )

    # Sort by channel
    for i, (k, v) in enumerate(items := sorted(transformed.items(), key=sort_key)):
        out[:, i] = v[:, crop:-crop:downsample, crop:-crop:downsample]
    del transformed

    # out[0, -1] = fids[reference][crop:-crop:downsample, crop:-crop:downsample]

    keys: list[str] = [k for k, _ in items]
    logger.debug(str([f"{i}: {k}" for i, k in enumerate(keys, 1)]))

    # (path / "down2").mkdir(exist_ok=True)

    # Use configured compression settings for final output
    compression_level = context.stitching_config.compression_levels.get("high", 0.75)
    tifffile.imwrite(
        out_path / f"reg-{idx:04d}.tif",
        out,
        compression=22610,
        compressionargs={"level": compression_level},
        metadata={
            "key": keys,
            "axes": "ZCYX",
            "shifts": json.dumps(shifts, cls=NumpyEncoder),
            "config": json.dumps(config.model_dump(), cls=NumpyEncoder),
        },
    )


def create_configuration(
    config_file: Path | None = None, data_path: str | None = None, **cli_overrides: Any
) -> Config:
    """Create configuration with proper precedence: CLI > Config File > Defaults.

    Args:
        config_file: Optional TOML configuration file path
        data_path: Optional data directory path (defaults to workspace-relative or fallback)
        **cli_overrides: CLI parameter overrides (fwhm, threshold, reference, etc.)

    Returns:
        Config object with proper parameter precedence
    """
    # Provide intelligent data_path default if not specified
    if data_path is None:
        # Try common locations in order of preference
        candidate_paths = [
            Path.cwd() / "data",  # Current working directory
            Path("/working/fishtools/data"),  # Default fallback
        ]

        for candidate in candidate_paths:
            if candidate.exists():
                data_path = str(candidate)
                break
        else:
            # Use fallback even if it doesn't exist
            data_path = str(candidate_paths[-1])
            logger.warning(f"Data directory not found, using fallback: {data_path}")

    if config_file and config_file.exists():
        logger.info(f"Loading configuration from {config_file}")
        try:
            return load_config_from_toml(config_file, data_path, **cli_overrides)
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            logger.info("Falling back to minimal configuration")

    # Backward compatibility path - use minimal config with CLI overrides
    logger.debug("Using minimal configuration with CLI parameters")
    return load_minimal_config(data_path=data_path, **cli_overrides)


console = Console()
app = typer.Typer(name="register", help="🔬 FISH Image Registration")


def scan_workspace_once(
    workspace_path: Path, reference: str = "2_10_18"
) -> dict[str, list[int]]:
    """Single filesystem scan to find all available work."""
    pattern = f"{reference}--*/{reference}*.tif"
    files = list(workspace_path.rglob(pattern))

    roi_files: dict[str, list[int]] = {}
    for file_path in files:
        try:
            # Extract ROI from path: reference--roi_name/file.tif
            roi_dir = file_path.parent.name  # e.g., "2_10_18--cortex"
            roi = roi_dir.split("--", 1)[1]  # Extract "cortex"

            # Extract index from filename: reference-0042.tif -> 42
            idx = int(file_path.stem.split("-")[1])
            roi_files.setdefault(roi, []).append(idx)

        except (ValueError, IndexError):
            # Skip malformed files silently
            continue

    # Return sorted indices for reproducible results
    return {roi: sorted(indices) for roi, indices in roi_files.items()}


def process_files(
    workspace_path: Path, target_files: list[tuple[str, int]], **kwargs
) -> None:
    """Process files with appropriate parallelism."""
    if len(target_files) == 1:
        # Single file - just do it
        roi, idx = target_files[0]
        try:
            _run(path=workspace_path, roi=roi, idx=idx, **kwargs)
            console.print(f"[green]✓ Processed {roi}:{idx}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed ROI: {roi} Idx:{idx}: {e}[/red]")
            console.print_exception()
        return

    # Multiple files - use thread pool
    def process_one(roi: str, idx: int) -> bool:
        try:
            _run(path=workspace_path, roi=roi, idx=idx, **kwargs)
            return True
        except Exception as e:
            console.print(f"[red]✗ Failed {roi}:{idx}: {e}[/red]")
            return False

    results = []
    # Use configured thread count for register operations
    thread_count = 4  # fallback
    try:
        # Get config-based thread count if available
        config = kwargs.get("config")
        if (
            config
            and hasattr(config, "registration")
            and hasattr(config.registration, "threads")
        ):
            thread_count = config.registration.threads
    except (AttributeError, KeyError):
        pass

    with progress_bar_threadpool(len(target_files), threads=thread_count) as submit:
        for roi, idx in target_files:
            future = submit(process_one, roi, idx)
            results.append(future)

    # Wait for completion and report
    successful = sum(result.result() for result in results)
    total = len(target_files)

    if successful == total:
        console.print(f"[green]✓ All {successful} files processed successfully[/green]")
    else:
        failed = total - successful
        console.print(f"[yellow]⚠ {successful} succeeded, {failed} failed[/yellow]")
        raise typer.Exit(1)


@app.command()
def register(
    path: Annotated[Path, typer.Argument(help="Workspace directory")],
    roi: Annotated[str | None, typer.Argument(help="ROI name (optional)")] = None,
    idx: Annotated[
        int | None, typer.Argument(help="Image index (requires ROI)")
    ] = None,
    # Processing options
    codebook: Annotated[
        Path | None, typer.Option("--codebook", "-c", help="Codebook JSON file")
    ] = None,
    reference: Annotated[
        str, typer.Option("--reference", help="Reference frame identifier")
    ] = "2_10_18",  # Aligned with config default
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Fiducial threshold")
    ] = 3.0,  # Aligned with config default
    fwhm: Annotated[
        float, typer.Option("--fwhm", "-f", help="FWHM for detection")
    ] = 4.0,  # Aligned with config default
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Enable debug logging")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", "-o", help="Overwrite existing")
    ] = False,
    no_priors: Annotated[
        bool, typer.Option("--no-priors", help="Disable priors")
    ] = False,
    config_file: Annotated[
        Path | None, typer.Option("--config", help="TOML config file")
    ] = None,
) -> None:
    """
    Register FISH images with progressive scoping: path [roi] [idx]

    Examples:
      register /workspace              # All files in all ROIs
      register /workspace cortex       # All files in cortex ROI
      register /workspace cortex 42    # File 42 in cortex ROI
    """

    # Basic validation
    if not path.exists():
        raise typer.BadParameter(f"Directory not found: {path}")

    if idx is not None and roi is None:
        raise typer.BadParameter(
            "Index requires ROI. Usage: register <path> <roi> <idx>"
        )

    if codebook and not codebook.exists():
        raise typer.BadParameter(f"Codebook not found: {codebook}")

    # Initialize logging if requested
    if debug:
        initialize_logger(idx=idx or 0, debug=debug)

    # Single scan to understand available work
    all_files = scan_workspace_once(path, reference)
    if not all_files:
        console.print(f"[yellow]⚠ No files found with reference '{reference}'[/yellow]")
        return

    # Determine target files based on arguments
    if roi is None:
        # All files in all ROIs
        target_files = [(r, i) for r, indices in all_files.items() for i in indices]
        console.print(
            f"[blue]📂 Processing {len(target_files)} files across {len(all_files)} ROIs[/blue]"
        )

    elif idx is None:
        # All files in specific ROI
        if roi not in all_files:
            available = list(all_files.keys())[:5]
            raise typer.BadParameter(
                f"ROI '{roi}' not found.\n"
                f"Available ROIs: {', '.join(available)}{'...' if len(all_files) > 5 else ''}"
            )
        target_files = [(roi, i) for i in all_files[roi]]
        console.print(
            f"[blue]📂 Processing {len(target_files)} files in ROI '{roi}'[/blue]"
        )

    else:
        # Specific file in specific ROI
        if roi not in all_files:
            available = list(all_files.keys())[:5]
            raise typer.BadParameter(
                f"ROI '{roi}' not found.\n"
                f"Available ROIs: {', '.join(available)}{'...' if len(all_files) > 5 else ''}"
            )

        if idx not in all_files[roi]:
            available_indices = all_files[roi][:10]
            suggestion = available_indices[0] if available_indices else "N/A"
            raise typer.BadParameter(
                f"Index {idx} not found in ROI '{roi}'.\n"
                f"Available indices: {', '.join(map(str, available_indices))}{'...' if len(all_files[roi]) > 10 else ''}\n"
                f"Try: register {path} {roi} {suggestion}"
            )

        target_files = [(roi, idx)]
        console.print(f"[blue]🎯 Processing specific file: {roi}:{idx}[/blue]")

    # Create configuration
    config = create_configuration(
        config_file=config_file, fwhm=fwhm, threshold=threshold
    )

    # Common processing arguments
    kwargs = {
        "codebook": codebook,
        "config": config,
        "debug": debug,
        "reference": reference,
        "overwrite": overwrite,
        "no_priors": no_priors,
    }

    # Process the files
    process_files(path, target_files, **kwargs)


# Old Click batch command removed - functionality replaced by unified Typer command above

if __name__ == "__main__":
    app()

# %%
