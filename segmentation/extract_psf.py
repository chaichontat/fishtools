# pip install numpy scipy scikit-image
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import difference_of_gaussians
from skimage.registration import phase_cross_correlation
from tifffile import imread, imwrite


@dataclass
class PSFResult:
    """
    Results of PSF estimation from bead stacks.
    Shapes use the same axis convention as the input stack (Z, C, Y, X) where applicable.
    """

    psf_zcyx: np.ndarray  # (Zp, C, Yp, Xp) averaged PSF (sum-normalized per channel)
    psf_peaknorm_zcyx: np.ndarray  # (Zp, C, Yp, Xp) peak-normalized per channel [max=1]
    bead_centers_z_y_x: np.ndarray  # (N, 3) bead centers detected in the reference channel (z,y,x)
    rz_by_channel: Dict[int, np.ndarray]  # channel -> (Zp, Nr) radius-vs-Z image (mean over theta)
    r_bins: np.ndarray  # (Nr,) centers of radial bins [pixels]
    params: Dict  # parameters used


def _ensure_odd(size: int) -> int:
    return int(size) if int(size) % 2 == 1 else int(size) + 1


def detect_beads_3d(
    vol_z_y_x: np.ndarray,
    dog_low_sigma: float = 0.8,
    dog_high_sigma: float = 3.0,
    threshold_rel: float = 0.2,
    min_distance: int = 6,
    n_peaks_max: Optional[int] = 500,
    valid_margin_zyx: Tuple[int, int, int] = (8, 8, 8),
) -> np.ndarray:
    """
    3D bead detection on a single-channel volume (Z,Y,X).
    Returns integer coordinates (z,y,x) of candidate beads.

    Strategy: DoG -> 3D local maxima -> edge/margin filtering -> keep strongest.
    """
    vol = vol_z_y_x.astype(np.float32, copy=False)
    dog = difference_of_gaussians(vol, dog_low_sigma, dog_high_sigma)
    # Absolute threshold from relative to peak (robust to scale)
    thr = threshold_rel * float(dog.max() if np.isfinite(dog.max()) else 0.0)
    logger.debug(
        "detect_beads_3d: low_sigma={low}, high_sigma={high}, thr={thr:.4f}, min_distance={dist}",
        low=dog_low_sigma,
        high=dog_high_sigma,
        thr=thr,
        dist=min_distance,
    )

    coords = peak_local_max(
        dog,
        threshold_abs=thr,
        min_distance=min_distance,
        exclude_border=False,
        footprint=None,
    )

    if coords.size == 0:
        return coords  # empty

    # Drop candidates too close to edges to allow full ROI extraction later.
    zmar, ymar, xmar = valid_margin_zyx
    Z, Y, X = vol.shape
    keep = (
        (coords[:, 0] >= zmar)
        & (coords[:, 0] < Z - zmar)
        & (coords[:, 1] >= ymar)
        & (coords[:, 1] < Y - ymar)
        & (coords[:, 2] >= xmar)
        & (coords[:, 2] < X - xmar)
    )
    coords = coords[keep]
    if coords.size == 0:
        return coords

    # Sort by DoG response (strongest first)
    strengths = dog[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(strengths)[::-1]
    coords = coords[order]
    if n_peaks_max is not None:
        coords = coords[:n_peaks_max]
    logger.debug("detect_beads_3d: kept {count} bead candidates", count=coords.shape[0])
    return coords


def extract_crops_zcyx(
    stack_zcyx: np.ndarray,
    centers_zyx: np.ndarray,
    roi_size_zyx: Tuple[int, int, int],
) -> List[np.ndarray]:
    """
    Extract Z×C×Y×X crops for each bead center. Returns list of arrays with shape (Zr, C, Yr, Xr).
    """
    Z, C, Y, X = stack_zcyx.shape
    wz, wy, wx = (_ensure_odd(s) for s in roi_size_zyx)
    rz, ry, rx = wz // 2, wy // 2, wx // 2

    crops: List[np.ndarray] = []
    for zc, yc, xc in centers_zyx:
        z0, z1 = zc - rz, zc + rz + 1
        y0, y1 = yc - ry, yc + ry + 1
        x0, x1 = xc - rx, xc + rx + 1
        if z0 < 0 or y0 < 0 or x0 < 0 or z1 > Z or y1 > Y or x1 > X:
            continue  # safety
        crops.append(stack_zcyx[z0:z1, :, y0:y1, x0:x1].astype(np.float32, copy=False))
    logger.debug(
        "extract_crops_zcyx: extracted {count} crops with roi={roi}",
        count=len(crops),
        roi=tuple(int(_ensure_odd(s)) for s in roi_size_zyx),
    )
    return crops


def subtract_local_background(
    crop_zcyx: np.ndarray,
    border: int = 2,
) -> np.ndarray:
    """
    Subtracts a robust background per channel using the median of the border voxels in the crop.
    """
    z, c, y, x = crop_zcyx.shape
    mask = np.ones((z, y, x), dtype=bool)
    mask[border : z - border, border : y - border, border : x - border] = False
    out = crop_zcyx.copy()
    for ch in range(c):
        bg = np.median(crop_zcyx[:, ch][mask])
        out[:, ch] -= bg
    logger.trace("subtract_local_background: applied border={border}", border=border)
    return out


def register_crops_subpixel(
    crops_zcyx: List[np.ndarray],
    ref_channel: int = 0,
    upsample_factor: int = 10,
) -> List[np.ndarray]:
    """
    Subpixel-register all crops to the first crop using phase cross correlation on the reference channel.
    The estimated (z,y,x) shift is applied to all channels within each crop.
    """
    if len(crops_zcyx) == 0:
        return []

    # Use the brightest/first crop as reference on the ref channel
    ref = crops_zcyx[0][:, ref_channel, :, :]
    aligned: List[np.ndarray] = []
    for idx, crop in enumerate(crops_zcyx):
        moving = crop[:, ref_channel, :, :]
        shift_zyx, _, _ = phase_cross_correlation(
            ref, moving, upsample_factor=upsample_factor, normalization=None
        )
        # Apply the same shift to all channels
        shifted = np.empty_like(crop)
        for ch in range(crop.shape[1]):
            shifted[:, ch] = ndi.shift(
                crop[:, ch],
                shift=shift_zyx,
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
        aligned.append(shifted)
        logger.trace(
            "register_crops_subpixel: crop {idx} registered with shift={shift}",
            idx=idx,
            shift=tuple(float(s) for s in shift_zyx),
        )
    logger.debug("register_crops_subpixel: registered {count} crops", count=len(aligned))
    return aligned


def robust_average_psf(crops_zcyx: List[np.ndarray], reduce: str = "median") -> np.ndarray:
    """
    Robustly average aligned crops into a PSF per channel.
    Returns array with shape (Zr, C, Yr, Xr).
    """
    stack = np.stack(crops_zcyx, axis=0)  # (N, Z, C, Y, X)
    if reduce == "mean":
        psf = np.mean(stack, axis=0)
    elif reduce == "median":
        psf = np.median(stack, axis=0)
    else:
        raise ValueError("reduce must be 'mean' or 'median'")
    logger.debug("robust_average_psf: reduced {n} crops with mode='{mode}'", n=stack.shape[0], mode=reduce)
    return psf


def normalize_psf_zcyx(psf_zcyx: np.ndarray, mode: str = "sum") -> np.ndarray:
    """
    Normalize per-channel either by total sum ('sum')—good for deconvolution—or by peak ('peak').
    """
    psf = psf_zcyx.copy()
    # loop over channels
    for ch in range(psf.shape[1]):
        v = psf[:, ch]
        if mode == "sum":
            s = v.sum()
            if s > 0:
                psf[:, ch] = v / s
        elif mode == "peak":
            m = v.max()
            if m > 0:
                psf[:, ch] = v / m
        else:
            raise ValueError("mode must be 'sum' or 'peak'")
    logger.debug("normalize_psf_zcyx: mode='{mode}'", mode=mode)
    return psf


def radial_average_xy_vs_z(
    vol_z_y_x: np.ndarray,
    center_yx: Optional[Tuple[float, float]] = None,
    dr: float = 1.0,
    r_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute azimuthal (theta) average for each Z plane -> (Z, Nr) where Nr is # radial bins.
    Returns (rz, r_centers). 'rz[z, rbin]' is the mean intensity at radius rbin for plane z.

    dr is in pixels; isotropic voxels assumed for Z,Y,X here.
    """
    Z, Y, X = vol_z_y_x.shape
    if center_yx is None:
        cy, cx = (Y - 1) / 2.0, (X - 1) / 2.0
    else:
        cy, cx = center_yx

    yy, xx = np.indices((Y, X))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    if r_max is None:
        r_max = float(min(cy, cx, Y - 1 - cy, X - 1 - cx))
    nbins = int(np.floor(r_max / dr)) + 1
    r_edges = np.arange(nbins + 1, dtype=np.float32) * dr
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    rz = np.zeros((Z, nbins), dtype=np.float32)
    for z in range(Z):
        plane = vol_z_y_x[z]
        # Mask to within r_max
        mask = rr < r_max
        rbin = np.floor(rr[mask] / dr).astype(int)
        vals = plane[mask].astype(np.float32)
        sums = np.bincount(rbin, weights=vals, minlength=nbins)
        counts = np.bincount(rbin, minlength=nbins)
        counts[counts == 0] = 1  # avoid divide-by-zero
        rz[z] = sums / counts
    return rz, r_centers


def estimate_psf_from_beads(
    stack_zcyx: np.ndarray,
    ref_channel: int = 0,
    roi_size_zyx: Tuple[int, int, int] = (21, 31, 31),  # (Z,Y,X); odd sizes enforced
    dog_low_sigma: float = 0.8,
    dog_high_sigma: float = 3.0,
    threshold_rel: float = 0.2,
    min_distance: int = 6,
    n_peaks_max: Optional[int] = 300,
    reduce: str = "median",
    background_border: int = 2,
    upsample_factor: int = 10,
    radial_dr: float = 1.0,
    radial_rmax: Optional[float] = None,
) -> PSFResult:
    """
    Full pipeline from Z×C×Y×X bead stack to per-channel PSF and radial (R×Z) representation.

    Notes:
      - Detection is done on the reference channel only.
      - One subpixel shift per bead (estimated on ref channel) is applied to all channels.
      - Output 'psf_zcyx' is sum-normalized per channel (also returns peak-normalized version).
      - Radial average is computed from the averaged (registered) PSF per channel.
    """
    assert stack_zcyx.ndim == 4 and stack_zcyx.shape[1] >= 1, "Expect (Z, C, Y, X)"
    Z, C, Y, X = stack_zcyx.shape
    logger.info(
        "estimate_psf_from_beads: stack shape=(Z={Z}, C={C}, Y={Y}, X={X}), ref_channel={ref}",
        Z=Z,
        C=C,
        Y=Y,
        X=X,
        ref=ref_channel,
    )
    # 1) Detect bead centers on the reference channel
    ref_vol = stack_zcyx[:, ref_channel, :, :]
    # Safety normalization for detection
    if ref_vol.max() > 0:
        ref_vol = (ref_vol / ref_vol.max()).astype(np.float32, copy=False)
    centers = detect_beads_3d(
        ref_vol,
        dog_low_sigma=dog_low_sigma,
        dog_high_sigma=dog_high_sigma,
        threshold_rel=threshold_rel,
        min_distance=min_distance,
        n_peaks_max=n_peaks_max,
        valid_margin_zyx=tuple(_ensure_odd(s) // 2 + 1 for s in roi_size_zyx),
    )
    if centers.size == 0:
        raise RuntimeError("No beads detected. Consider lowering threshold_rel or min_distance.")
    logger.info("estimate_psf_from_beads: detected {count} bead centers", count=centers.shape[0])

    # 2) Extract crops around each bead (shared across channels)
    crops = extract_crops_zcyx(stack_zcyx, centers, roi_size_zyx=roi_size_zyx)
    if len(crops) == 0:
        raise RuntimeError("All detected beads were too close to borders for the chosen ROI size.")
    logger.info("estimate_psf_from_beads: extracted {count} valid crops", count=len(crops))

    # 3) Background subtraction (robust, per crop & channel)
    crops = [subtract_local_background(c, border=background_border) for c in crops]
    logger.debug(
        "estimate_psf_from_beads: background subtraction applied with border={border}",
        border=background_border,
    )

    # 4) Subpixel registration of crops using the reference channel
    aligned = register_crops_subpixel(crops, ref_channel=ref_channel, upsample_factor=upsample_factor)
    logger.info("estimate_psf_from_beads: registered {count} crops", count=len(aligned))

    # 5) Robust averaging into a PSF per channel
    psf = robust_average_psf(aligned, reduce=reduce)  # (Zr, C, Yr, Xr)
    logger.info("estimate_psf_from_beads: averaged PSF shape={shape}", shape=tuple(int(s) for s in psf.shape))

    # 6) Normalizations
    psf_sum = normalize_psf_zcyx(psf, mode="sum")
    psf_peak = normalize_psf_zcyx(psf, mode="peak")
    logger.debug("estimate_psf_from_beads: normalization complete")

    # 7) Radially symmetrized (R×Z) images per channel from the averaged PSF
    rz_by_ch: Dict[int, np.ndarray] = {}
    r_bins_out: Optional[np.ndarray] = None
    for ch in range(C):
        rz, r_bins = radial_average_xy_vs_z(
            psf_sum[:, ch, :, :],
            center_yx=None,  # crops are centered; registration makes COM near center
            dr=radial_dr,
            r_max=radial_rmax,
        )
        rz_by_ch[ch] = rz  # (Zr, Nr)
        if r_bins_out is None:
            r_bins_out = r_bins
    assert r_bins_out is not None
    logger.info(
        "estimate_psf_from_beads: computed radial profiles for {channels} channels with {bins} bins",
        channels=len(rz_by_ch),
        bins=len(r_bins_out),
    )

    return PSFResult(
        psf_zcyx=psf_sum,
        psf_peaknorm_zcyx=psf_peak,
        bead_centers_z_y_x=centers,
        rz_by_channel=rz_by_ch,
        r_bins=r_bins_out,
        params=dict(
            ref_channel=ref_channel,
            roi_size_zyx=tuple(int(_ensure_odd(s)) for s in roi_size_zyx),
            dog_low_sigma=dog_low_sigma,
            dog_high_sigma=dog_high_sigma,
            threshold_rel=threshold_rel,
            min_distance=min_distance,
            n_peaks_max=n_peaks_max,
            reduce=reduce,
            background_border=background_border,
            upsample_factor=upsample_factor,
            radial_dr=radial_dr,
            radial_rmax=radial_rmax,
        ),
    )


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def save_psf_outputs(
    outdir: str,
    basename: str,
    res,
    save_tiff: bool = True,
    save_npy: bool = True,
):
    """
    Save per-file outputs:
      - psf_sum (Z,C,Y,X) per channel to multi-page TIFF and per-channel TIFFs
      - peak-normalized PSF to TIFF
      - R×Z arrays to .npy (and optional CSV)
      - metadata JSON (parameters, centers)
    """
    _ensure_dir(outdir)
    psf_sum = res.psf_zcyx  # (Z,C,Y,X)
    psf_peak = res.psf_peaknorm_zcyx
    r_bins = res.r_bins
    rz_by_ch: Dict[int, np.ndarray] = res.rz_by_channel  # ch -> (Z,Nr)
    logger.info(
        "save_psf_outputs: writing outputs for {basename} to {outdir}",
        basename=basename,
        outdir=outdir,
    )
    saved_paths: list[str] = []

    if save_tiff:
        # full stack per type
        path_sum = os.path.join(outdir, f"{basename}__psf_sum.tif")
        imwrite(path_sum, psf_sum.astype(np.float32))
        saved_paths.append(path_sum)
        path_peak = os.path.join(outdir, f"{basename}__psf_peak.tif")
        imwrite(path_peak, psf_peak.astype(np.float32))
        saved_paths.append(path_peak)
        # per-channel convenience
        for ch in range(psf_sum.shape[1]):
            path_ch_sum = os.path.join(outdir, f"{basename}__psf_sum_ch{ch}.tif")
            imwrite(path_ch_sum, psf_sum[:, ch].astype(np.float32))
            saved_paths.append(path_ch_sum)
            path_ch_peak = os.path.join(outdir, f"{basename}__psf_peak_ch{ch}.tif")
            imwrite(path_ch_peak, psf_peak[:, ch].astype(np.float32))
            saved_paths.append(path_ch_peak)
            channel_stack = psf_sum[:, ch].astype(np.float32)
            projections: Dict[str, np.ndarray] = {}
            for axis_name, axis_idx in (("z", 0), ("y", 1), ("x", 2)):
                proj = np.max(channel_stack, axis=axis_idx)
                proj_path = os.path.join(outdir, f"{basename}__psf_sum_ch{ch}_maxproj_{axis_name}.tif")
                imwrite(proj_path, proj.astype(np.float32))
                saved_paths.append(proj_path)
                projections[axis_name] = proj.astype(np.float32)

            # Build summary figure with radial profile and projections
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
            radial = rz_by_ch.get(ch)
            if radial is not None:
                im = axes[0, 0].imshow(radial, aspect="auto", origin="lower", cmap="magma")
                axes[0, 0].set_title("Radial profile (Z × r)")
                axes[0, 0].set_xlabel("Radius bin")
                axes[0, 0].set_ylabel("Z index")
                fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
            else:
                axes[0, 0].set_visible(False)

            axes_map = {
                (0, 1): ("Max proj (Z axis)", projections.get("z"), ("X", "Y")),
                (1, 0): ("Max proj (Y axis)", projections.get("y"), ("X", "Z")),
                (1, 1): ("Max proj (X axis)", projections.get("x"), ("Y", "Z")),
            }
            for (row, col), (title, data, labels) in axes_map.items():
                ax = axes[row, col]
                if data is None:
                    ax.set_visible(False)
                    continue
                im = ax.imshow(data, origin="lower", cmap="viridis")
                ax.set_title(title)
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig_path = os.path.join(outdir, f"{basename}__summary_ch{ch}.png")
            fig.suptitle(f"{basename} – channel {ch}")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            saved_paths.append(fig_path)

    if save_npy:
        path_sum_npy = os.path.join(outdir, f"{basename}__psf_sum.npy")
        np.save(path_sum_npy, psf_sum)
        saved_paths.append(path_sum_npy)
        path_peak_npy = os.path.join(outdir, f"{basename}__psf_peak.npy")
        np.save(path_peak_npy, psf_peak)
        saved_paths.append(path_peak_npy)
        path_rbins = os.path.join(outdir, f"{basename}__r_bins.npy")
        np.save(path_rbins, r_bins)
        saved_paths.append(path_rbins)
        for ch, rz in rz_by_ch.items():
            rz_path = os.path.join(outdir, f"{basename}__rz_ch{ch}.npy")
            np.save(rz_path, rz.astype(np.float32))
            saved_paths.append(rz_path)
            rz_max_z = np.max(rz, axis=0).astype(np.float32)
            rz_max_r = np.max(rz, axis=1).astype(np.float32)
            rz_max_z_path = os.path.join(outdir, f"{basename}__rz_ch{ch}_maxproj_z.npy")
            np.save(rz_max_z_path, rz_max_z)
            saved_paths.append(rz_max_z_path)
            rz_max_r_path = os.path.join(outdir, f"{basename}__rz_ch{ch}_maxproj_r.npy")
            np.save(rz_max_r_path, rz_max_r)
            saved_paths.append(rz_max_r_path)

    meta = {
        "basename": basename,
        "params": res.params,
        "n_beads_detected": int(res.bead_centers_z_y_x.shape[0]),
        "bead_centers_zyx": res.bead_centers_z_y_x.tolist(),
        "r_bins_len": int(len(r_bins)),
        "shape_psf": list(psf_sum.shape),
    }
    meta_path = os.path.join(outdir, f"{basename}__meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.debug("save_psf_outputs: wrote metadata with {n_beads} beads", n_beads=meta["n_beads_detected"])
    saved_paths.append(meta_path)

    if saved_paths:
        logger.info(
            "save_psf_outputs: saved {count} artifacts for {basename}; last path: {last}",
            count=len(saved_paths),
            basename=basename,
            last=saved_paths[-1],
        )
        for path in saved_paths:
            logger.debug("save_psf_outputs: saved path {path}", path=path)
    else:
        logger.warning("save_psf_outputs: no artifacts were saved for {basename}", basename=basename)
    return saved_paths


def batch_estimate_psf(
    input_glob: str,
    output_dir: str,
    ref_channel: int = 0,
    roi_size_zyx=(21, 31, 31),
    threshold_rel: float = 0.15,
    min_distance: int = 8,
    n_peaks_max: int = 400,
    reduce: str = "median",
    upsample_factor: int = 10,
    radial_dr: float = 1.0,
    radial_rmax: float | None = None,
    save_intermediate: bool = True,
):
    """
    Process many Z×C×Y×X files that match input_glob.
    Produces per-file outputs + a global, cross-file PSF/peak and R×Z per channel.
    """
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_glob}")
    logger.info("batch_estimate_psf: matched {n} files for glob '{glob}'", n=len(files), glob=input_glob)

    _ensure_dir(output_dir)
    perfile_psfs: List[np.ndarray] = []  # each (Z,C,Y,X)
    perfile_rz: Dict[int, List[np.ndarray]] = {}  # ch -> list[(Z,Nr)]
    r_bins_global: np.ndarray | None = None

    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        logger.info("batch_estimate_psf: processing {base}", base=base)
        arr = imread(f)

        if arr.ndim == 3:
            trimmed = arr[:-2]
            try:
                arr = trimmed.reshape(50, 3, 2048, 2048)
            except ValueError as exc:
                raise ValueError(
                    f"{f}: failed to reshape 3D input {trimmed.shape} into (50, 3, 2048, 2048)"
                ) from exc
            logger.debug(
                "batch_estimate_psf: reshaped ZYX input {shape} to (50, 3, 2048, 2048)", shape=trimmed.shape
            )
        # Expect Z×C×Y×X; if you know they're large-endian or need dtype cast, do it here.
        if arr.ndim != 4:
            raise ValueError(f"{f}: expected 4D (Z,C,Y,X), got {arr.shape}")
        Z, C, Y, X = arr.shape
        if C < 1:
            raise ValueError(f"{f}: C dimension must be >=1")

        res = estimate_psf_from_beads(
            stack_zcyx=arr,
            ref_channel=ref_channel,
            roi_size_zyx=roi_size_zyx,
            threshold_rel=threshold_rel,
            min_distance=min_distance,
            n_peaks_max=n_peaks_max,
            reduce=reduce,
            upsample_factor=upsample_factor,
            radial_dr=radial_dr,
            radial_rmax=radial_rmax,
        )

        if save_intermediate:
            saved_paths = save_psf_outputs(os.path.join(output_dir, "per_file"), base, res)
            if saved_paths:
                logger.info(
                    "batch_estimate_psf: per-file outputs saved ({count} artifacts); first path: {first}",
                    count=len(saved_paths),
                    first=saved_paths[0],
                )

        perfile_psfs.append(res.psf_zcyx.astype(np.float32))  # already sum-normalized per channel
        # collect R×Z per channel
        if r_bins_global is None:
            r_bins_global = res.r_bins
        else:
            # sanity: all runs should share the same binning if roi and dr unchanged
            if len(res.r_bins) != len(r_bins_global) or not np.allclose(res.r_bins, r_bins_global):
                raise RuntimeError("R-bin definitions differ across files; fix dr/r_max/ROI.")

        for ch, rz in res.rz_by_channel.items():
            perfile_rz.setdefault(ch, []).append(rz.astype(np.float32))

    # --- Aggregate across files ---
    logger.info("batch_estimate_psf: aggregating global PSF across {n} files", n=len(perfile_psfs))
    stack_psf = np.stack(perfile_psfs, axis=0)  # (N, Z, C, Y, X)
    global_psf_median = np.median(stack_psf, axis=0)  # (Z, C, Y, X)
    global_psf_mean = np.mean(stack_psf, axis=0)  # optional

    # peak-normalized copy for viz
    global_psf_peak = global_psf_median.copy()
    for ch in range(global_psf_peak.shape[1]):
        v = global_psf_peak[:, ch]
        m = v.max()
        if m > 0:
            global_psf_peak[:, ch] = v / m

    # Aggregate R×Z per channel
    global_rz_by_ch: Dict[int, Dict[str, np.ndarray]] = {}
    for ch, lst in perfile_rz.items():
        rz_stack = np.stack(lst, axis=0)  # (N, Z, Nr)
        global_rz_by_ch[ch] = {
            "median": np.median(rz_stack, axis=0),
            "mean": np.mean(rz_stack, axis=0),
        }

    # --- Save global outputs ---
    gdir = _ensure_dir(os.path.join(output_dir, "global"))
    global_saved_paths: list[str] = []
    global_psf_median_tif = os.path.join(gdir, "global_psf_median.tif")
    imwrite(global_psf_median_tif, global_psf_median.astype(np.float32))
    global_saved_paths.append(global_psf_median_tif)
    global_psf_peak_tif = os.path.join(gdir, "global_psf_peak.tif")
    imwrite(global_psf_peak_tif, global_psf_peak.astype(np.float32))
    global_saved_paths.append(global_psf_peak_tif)
    global_psf_median_npy = os.path.join(gdir, "global_psf_median.npy")
    np.save(global_psf_median_npy, global_psf_median)
    global_saved_paths.append(global_psf_median_npy)
    global_psf_mean_npy = os.path.join(gdir, "global_psf_mean.npy")
    np.save(global_psf_mean_npy, global_psf_mean)
    global_saved_paths.append(global_psf_mean_npy)
    r_bins_path = os.path.join(gdir, "r_bins.npy")
    np.save(r_bins_path, r_bins_global)
    global_saved_paths.append(r_bins_path)
    for ch, d in global_rz_by_ch.items():
        rz_median_path = os.path.join(gdir, f"global_rz_ch{ch}_median.npy")
        np.save(rz_median_path, d["median"])
        global_saved_paths.append(rz_median_path)
        rz_mean_path = os.path.join(gdir, f"global_rz_ch{ch}_mean.npy")
        np.save(rz_mean_path, d["mean"])
        global_saved_paths.append(rz_mean_path)

    # small manifest
    manifest = {
        "n_files": len(files),
        "files": [os.path.basename(f) for f in files],
        "roi_size_zyx": list(roi_size_zyx),
        "ref_channel": ref_channel,
        "radial_dr": radial_dr,
        "r_bins_len": int(len(r_bins_global)),
        "global_psf_shape": list(global_psf_median.shape),
    }
    manifest_path = os.path.join(gdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    global_saved_paths.append(manifest_path)
    logger.info(
        "batch_estimate_psf: saved {count} global artifacts; output directory: {gdir}",
        count=len(global_saved_paths),
        gdir=gdir,
    )
    for path in global_saved_paths:
        logger.debug("batch_estimate_psf: global artifact path {path}", path=path)

    logger.info("batch_estimate_psf: finished, outputs in {output_dir}", output_dir=output_dir)
    return {
        "global_psf_median": global_psf_median,
        "global_psf_peak": global_psf_peak,
        "global_rz_by_channel": global_rz_by_ch,
        "r_bins": r_bins_global,
    }


if __name__ == "__main__":
    # img = imread("/warm/analyzed/20250725_beads/650--beads/650-0012.tif")[:-2].reshape(50, 3, 2048, 2048)
    # paths = save_psf_outputs(
    #     outdir="/warm/analyzed/20250725_beads/650--beads/psf_test",
    #     basename="650-0012",
    #     res=estimate_psf_from_beads(img, roi_size_zyx=(31, 31, 31)),
    # )
    # for p in paths:
    #     logger.info(f"Saved: {p}")
    # Example:
    # Example:
    # python batch_psf.py  (after placing this code in a file and pasting the earlier functions above)
    outputs = batch_estimate_psf(
        input_glob="/warm/analyzed/20250725_beads/650--beads/*.tif",  # each must be Z×C×Y×X
        output_dir="/warm/analyzed/20250725_beads/650--beads/psf_test",
        ref_channel=0,
        roi_size_zyx=(31, 35, 35),
        threshold_rel=0.15,
        min_distance=8,
        n_peaks_max=500,
        reduce="median",
        upsample_factor=10,
        radial_dr=1.0,
        radial_rmax=None,  # auto from ROI
        save_intermediate=True,
    )
