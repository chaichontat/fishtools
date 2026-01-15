from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import ants
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from fishtools.io.workspace import Workspace

# %% # === Configuration ===

# Default path (can be overridden by CLI)
DEFAULT_WORKSPACE = "/working/20251228_JaxA4_Sag4"
DEFAULT_ROI = "3"

DOWNSAMPLE_FACTOR = 1
FPS = 30
DURATION_PHASE_1 = 3.0  # seconds (Affine)
DURATION_PHASE_2 = 3.0  # seconds (Warp)
DURATION_PAUSE = 0.05   # seconds (Pause between phases)

MIN_WIDTH = 1280
MIN_HEIGHT = 720

def ants_numpy_yx(img: ants.ANTsImage) -> np.ndarray:
    """Return a 2D ANTs image as a NumPy array in (Y, X) order for display."""
    arr_xy = np.asarray(img.numpy())
    if arr_xy.ndim != 2:
        raise ValueError(f"Expected 2D ANTsImage, got shape={arr_xy.shape}.")
    return arr_xy.T


def interpolate_affine_params(params_start: np.ndarray, params_end: np.ndarray, t: float) -> list[float]:
    """Linearly interpolate affine parameters."""
    return (params_start + t * (params_end - params_start)).tolist()

def read_affine_params(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read affine parameters and fixed parameters from an ANTs/ITK .mat transform file."""
    tx = ants.read_transform(str(path))
    return np.asarray(tx.parameters, dtype=float), np.asarray(tx.fixed_parameters, dtype=float)


def write_affine_transform_file(*, path: Path, parameters: list[float], fixed_parameters: list[float]) -> None:
    """Write an affine transform (.mat) via ANTsPy (format matches ants.read_transform/apply_transforms)."""
    tx = ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=2,
        parameters=parameters,
        fixed_parameters=fixed_parameters,
    )
    ants.write_transform(tx, str(path))

def overlay_images(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    *,
    out_hw: tuple[int, int] | None = None,
    edge_dilate: int = 0,
) -> np.ndarray:
    """Create a visualization overlay (magenta edges fixed, green moving)."""
    # Normalize to 0-255
    def norm(x):
        x = x.astype(np.float32)
        mn, mx = np.percentile(x, (1, 99))
        x = np.clip((x - mn) / (mx - mn + 1e-6), 0, 1)
        return (x * 255).astype(np.uint8)

    f0 = norm(fixed_img)
    m0 = norm(moving_img)

    if out_hw is not None:
        out_h, out_w = out_hw
        src_h, src_w = f0.shape
        if out_h < src_h or out_w < src_w:
            raise ValueError(
                f"Refusing to shrink frames (would downsample): src={src_w}x{src_h}, out={out_w}x{out_h}."
            )
        if (out_h, out_w) == (src_h, src_w):
            f = f0
            m = m0
        else:
            f = cv2.resize(f0, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m0, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    else:
        f = f0
        m = m0
    
    # Edge detection for fixed image
    edges = cv2.Canny(f, 50, 150)
    if edge_dilate > 0:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=edge_dilate)
    
    # RGB image
    # Red: Fixed Edges
    # Green: Moving Image
    # Blue: Fixed Edges
    # Result: Magenta Edges on Green Image
    
    h, w = f.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    
    out[..., 0] = edges  # R
    out[..., 1] = m      # G
    out[..., 2] = edges  # B
    
    return out


def pad_to_min_size(rgb: np.ndarray, *, min_width: int, min_height: int) -> np.ndarray:
    """Pad an RGB image to at least (min_width, min_height) without resizing/downsampling."""
    h, w = rgb.shape[:2]
    out_h = max(h, min_height)
    out_w = max(w, min_width)
    if (out_h, out_w) == (h, w):
        return rgb

    top = (out_h - h) // 2
    bottom = out_h - h - top
    left = (out_w - w) // 2
    right = out_w - w - left
    return cv2.copyMakeBorder(rgb, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))


def find_font_path() -> Path:
    """Best-effort lookup for an Arial/Helvetica-like TTF/OTF font on Linux."""
    roots = (
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
    )
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if not (name.endswith(".ttf") or name.endswith(".otf")):
                continue
            if "arial" in name or "helvetica" in name:
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Could not find an Arial/Helvetica font file under /usr/share/fonts; "
            "pass --font-path to a .ttf/.otf (e.g. Arial.ttf)."
        )

    # Prefer Arial over Helvetica if both exist; prefer .ttf over .otf.
    candidates.sort(key=lambda p: ("arial" not in p.name.lower(), p.suffix.lower() != ".ttf", str(p)))
    return candidates[0]


def draw_center_label(
    rgb: np.ndarray,
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    stroke_width: int = 4,
) -> np.ndarray:
    """Draw a centered label on an RGB frame using a TrueType font (Arial/Helvetica)."""
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    w, h = img.size
    x = (w - text_w) // 2
    y = (h - text_h) // 2

    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(img)


def extract_warp_from_composite(
    composite_path: Path,
    affine_path: Path,
    reference_img: ants.ANTsImage
) -> ants.ANTsImage:
    """
    Decompose Composite (C) = Affine (A) o Warp (W) => W = inv(A) o C.
    Returns the Warp field as a vector image.
    """
    print("Extracting Warp field from Composite...")
    
    # 1. Generate grid points in Fixed Physical Space
    origin = reference_img.origin
    spacing = reference_img.spacing
    shape = reference_img.shape
    
    # Grid in index space
    x_idx = np.arange(shape[0])
    y_idx = np.arange(shape[1])
    xx, yy = np.meshgrid(x_idx, y_idx, indexing='ij') # 'ij' for ants/numpy consistency?
    # ANTs images: dim 0 is x, dim 1 is y in physical space usually?
    # But numpy array from ants image is (H, W)?
    # Let's trust ANTs physical coordinates.
    
    # Physical coordinates
    # p = origin + idx * spacing
    pts_x = origin[0] + xx * spacing[0]
    pts_y = origin[1] + yy * spacing[1]
    
    flat_x = pts_x.ravel()
    flat_y = pts_y.ravel()
    
    df_points = pd.DataFrame({"x": flat_x, "y": flat_y})
    
    # 2. Apply Composite: x' = C(x)
    # This gives the Moving coordinate that corresponds to x
    df_mapped = ants.apply_transforms_to_points(
        dim=2,
        points=df_points,
        transformlist=[str(composite_path)]
    )
    
    # 3. Apply Inverse Affine: x'' = inv(A)(x')
    # This gives the coordinate after just the Warp: W(x) = inv(A)(C(x))
    df_warp_only = ants.apply_transforms_to_points(
        dim=2,
        points=df_mapped,
        transformlist=[str(affine_path)],
        whichtoinvert=[True]
    )
    
    # 4. Displacement = W(x) - x
    disp_x = df_warp_only['x'].values - flat_x
    disp_y = df_warp_only['y'].values - flat_y
    
    # 5. Reshape to image
    # Note: reshape needs to match meshgrid 'ij'
    disp_x_img = disp_x.reshape(shape)
    disp_y_img = disp_y.reshape(shape)
    
    # Create vector image
    # ANTsPy doesn't have a direct "from_numpy_vector" easy wrapper?
    # We can create a multichannel image.
    # Stack channels
    disp_arr = np.stack([disp_x_img, disp_y_img], axis=-1) # (H, W, 2)
    
    # Create ANTs image
    warp_img = ants.from_numpy(
        disp_arr,
        origin=origin,
        spacing=spacing,
        direction=reference_img.direction,
        has_components=True
    )
    
    return warp_img

def main():
    parser = argparse.ArgumentParser(description="Animate registration process.")
    parser.add_argument("--workspace", type=Path, default=Path(DEFAULT_WORKSPACE))
    parser.add_argument("--roi", type=str, default=DEFAULT_ROI)
    parser.add_argument("--downsample", type=int, default=DOWNSAMPLE_FACTOR, help="Downsample factor (1 = no downsample).")
    parser.add_argument("--output", type=Path, help="Output video path")
    parser.add_argument("--min-width", type=int, default=MIN_WIDTH, help="Pad output to at least this width (px).")
    parser.add_argument("--min-height", type=int, default=MIN_HEIGHT, help="Pad output to at least this height (px).")
    parser.add_argument("--edge-dilate", type=int, default=0, help="Edge outline thickness (0 = no dilation).")
    parser.add_argument("--affine-sec", type=float, default=DURATION_PHASE_1, help="Duration of affine phase (sec).")
    parser.add_argument("--syn-sec", type=float, default=DURATION_PHASE_2, help="Duration of SyN phase (sec).")
    parser.add_argument("--pause-sec", type=float, default=DURATION_PAUSE, help="Pause duration between phases (sec).")
    parser.add_argument("--font-path", type=Path, default=None, help="Path to a .ttf/.otf font (Arial/Helvetica).")
    parser.add_argument("--font-size", type=int, default=0, help="Font size in px (0 = auto).")
    args = parser.parse_args()

    ws = Workspace(args.workspace)
    roi_dir = ws.ccf_transforms(args.roi)
    landmark_syn_dir = roi_dir / "landmark_syn_mi"
    summary_json_path = landmark_syn_dir / "similarity_plus_syn_summary.json"

    if not summary_json_path.exists():
        print(f"Error: Summary JSON not found at {summary_json_path}")
        return

    summary = json.loads(summary_json_path.read_text())
    paths = summary["paths"]
    
    fixed_path = Path(paths["fixed_nifti"])
    moving_path = Path(paths["moving_nifti"])
    
    if not fixed_path.exists():
        fixed_path = landmark_syn_dir / Path(paths["fixed_nifti"]).name
        moving_path = landmark_syn_dir / Path(paths["moving_nifti"]).name

    # Locate Transforms
    affine_path = None
    composite_path = None
    
    # 1. Find Affine
    # Try the one listed in summary paths
    if "linear_init_mat" in paths and Path(paths["linear_init_mat"]).exists():
        affine_path = Path(paths["linear_init_mat"])
    elif (landmark_syn_dir / "init_affine_fixed2moving_from_p1.mat").exists():
         affine_path = landmark_syn_dir / "init_affine_fixed2moving_from_p1.mat"
         
    # 2. Find Composite / Warp
    fwd_transforms = summary.get("fwdtransforms", [])
    if isinstance(fwd_transforms, str):
        fwd_transforms = [fwd_transforms]
    for t in fwd_transforms:
        if not isinstance(t, str):
            continue
        if t.endswith("Composite.h5"):
            if Path(t).exists():
                composite_path = Path(t)
            elif (landmark_syn_dir / Path(t).name).exists():
                composite_path = landmark_syn_dir / Path(t).name
            break

    if not affine_path:
        print("Error: Could not find linear init affine transform (.mat).")
        return
    
    print(f"Affine: {affine_path}")
    print(f"Composite: {composite_path}")

    # Load Images
    print("Loading images...")
    fixed_full = ants.image_read(str(fixed_path))
    moving_full = ants.image_read(str(moving_path))

    # Downsample
    ds = args.downsample
    if ds < 1:
        raise ValueError(f"--downsample must be >= 1, got {ds}")
    if ds == 1:
        print("No downsampling (using full resolution).")
        fixed_ds = fixed_full
        moving_ds = moving_full
    else:
        print(f"Downsampling images by {ds}x...")
        fixed_ds = ants.resample_image(
            fixed_full,
            (int(fixed_full.shape[0] / ds), int(fixed_full.shape[1] / ds)),
            use_voxels=True,
        )
        moving_ds = ants.resample_image(
            moving_full,
            (int(moving_full.shape[0] / ds), int(moving_full.shape[1] / ds)),
            use_voxels=True,
        )

    # Prepare Warp
    warp_field_ds = None
    if composite_path:
        warp_field_ds = extract_warp_from_composite(composite_path, affine_path, fixed_ds)
    else:
        print("Warning: No composite transform found. Will only animate Affine.")

    # Parse Affine Params
    final_params, fixed_params = read_affine_params(affine_path)
    identity_params = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)

    # Setup Video
    output_path = args.output
    if not output_path:
        output_path = landmark_syn_dir / "registration_animation.mp4"
    
    arr_fixed = ants_numpy_yx(fixed_ds)
    src_h, src_w = arr_fixed.shape

    # No display resampling (no downsampling); only pad to reach a minimum canvas size (e.g. 720p).
    out_h = max(src_h, args.min_height)
    out_w = max(src_w, args.min_width)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(str(output_path), fourcc, FPS, (out_w, out_h))
    
    print(f"Recording to {output_path} ({out_w}x{out_h})")

    if args.font_size < 0:
        raise ValueError(f"--font-size must be >= 0, got {args.font_size}")
    font_size = args.font_size if args.font_size > 0 else max(18, out_h // 10)
    font_path = args.font_path if args.font_path is not None else find_font_path()
    if not font_path.exists():
        raise FileNotFoundError(f"--font-path does not exist: {font_path}")
    font = ImageFont.truetype(str(font_path), font_size)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_mat = temp_dir_path / "temp_affine.mat"
        temp_warp = temp_dir_path / "temp_warp.nii.gz"
        
        # === Phase 1: Affine ===
        if args.affine_sec < 0:
            raise ValueError(f"--affine-sec must be >= 0, got {args.affine_sec}")
        frames_p1 = max(1, int(round(args.affine_sec * FPS)))
        print("Rendering Phase 1 (Affine)...")
        for i in tqdm(range(frames_p1)):
            t = 1.0 if frames_p1 == 1 else i / (frames_p1 - 1)
            t_ease = t * t * (3 - 2 * t)
            
            curr_params = interpolate_affine_params(identity_params, final_params, t_ease)
            write_affine_transform_file(path=temp_mat, parameters=curr_params, fixed_parameters=fixed_params.tolist())
            
            warped = ants.apply_transforms(
                fixed=fixed_ds,
                moving=moving_ds,
                transformlist=[str(temp_mat)],
                interpolator='linear'
            )
            
            frame = overlay_images(
                arr_fixed,
                ants_numpy_yx(warped),
                edge_dilate=args.edge_dilate,
            )
            frame = pad_to_min_size(frame, min_width=args.min_width, min_height=args.min_height)
            frame = draw_center_label(frame, "Affine", font=font)
            out_vid.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # === Phase 1 Pause ===
        print("Pause...")
        if args.pause_sec < 0:
            raise ValueError(f"--pause-sec must be >= 0, got {args.pause_sec}")
        frames_pause = int(round(args.pause_sec * FPS))
        for _ in range(frames_pause):
            out_vid.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # === Phase 2: Warp ===
        if warp_field_ds:
            if args.syn_sec < 0:
                raise ValueError(f"--syn-sec must be >= 0, got {args.syn_sec}")
            frames_p2 = max(1, int(round(args.syn_sec * FPS)))
            print("Rendering Phase 2 (Warp)...")
            
            final_affine_path = temp_dir_path / "final_affine.mat"
            write_affine_transform_file(
                path=final_affine_path,
                parameters=final_params.tolist(),
                fixed_parameters=fixed_params.tolist(),
            )

            for i in tqdm(range(frames_p2)):
                t = 1.0 if frames_p2 == 1 else i / (frames_p2 - 1)
                t_ease = t * t * (3 - 2 * t)
                
                curr_warp = warp_field_ds * t_ease
                ants.image_write(curr_warp, str(temp_warp))
                
                # Apply: [Warp, Affine]
                warped = ants.apply_transforms(
                    fixed=fixed_ds,
                    moving=moving_ds,
                    transformlist=[str(temp_warp), str(final_affine_path)],
                    interpolator='linear'
                )
                
                frame = overlay_images(
                    arr_fixed,
                    ants_numpy_yx(warped),
                    edge_dilate=args.edge_dilate,
                )
                frame = pad_to_min_size(frame, min_width=args.min_width, min_height=args.min_height)
                frame = draw_center_label(frame, "SyN", font=font)
                out_vid.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            # === Phase 2 Pause ===
            print("Final Pause...")
            for _ in range(frames_pause):
                out_vid.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out_vid.release()
    print("Done.")

if __name__ == "__main__":
    main()
