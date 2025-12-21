# Salvaging High-Shift Rounds

This document describes the workflow for recovering rounds with large drift offsets
(>100 pixels) that exceed the standard alignment tolerance. These rounds would normally
fail registration but can be salvaged by pre-stitching independently, applying a global
shift, and slicing back into tiles for the standard pipeline.

## Rationale

### The Problem

Standard inter-round registration assumes small drifts (<100px) that can be corrected
by the alignment algorithm. When drift exceeds this tolerance:

- Alignment fails or produces incorrect shifts
- The round cannot be registered with other rounds
- Valuable data is lost

### Why Not Just Shift Each Tile?

The naive solution would be to apply a large crop/pad shift to each tile individually.
This fails because:

- **Stitching requires margins** — The fusion algorithm needs ~40px overlap on each
  tile edge for proper blending
- **Shifted tiles have dark edges** — Cropping one side and zero-padding the other
  creates a black band on each tile
- **Dark bands propagate** — Since all tiles shift the same direction, all have dark
  edges on the same side. Neighboring tiles can't fill the gap because their overlap
  region is also dark. Result: dark bands throughout the mosaic.

### The Solution: Pre-Stitch, Shift, Slice

Instead, we stitch first, then shift the mosaic:

1. **Pre-stitch the problematic round** — Fuse all tiles using their original positions
   (accurate within the round). Tiles blend normally with no dark edges.

2. **Apply global shift to mosaic** — Crop/pad the entire stitched image. Dark edges
   appear only at the mosaic perimeter, not at interior tile boundaries.

3. **Slice back into tiles** — Cut the shifted mosaic into tiles matching the reference
   grid, producing images that feed back into the standard pipeline.

This works because:

- **Within-round positions are accurate** — Stage drift affects the global offset
  between rounds, not tile-to-tile positions within a round
- **Interior stays clean** — Dark edges only at mosaic boundary, not between tiles
- **Integer shifts preserve pixels** — No interpolation artifacts

---

## Workflow Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         Standard Pipeline                                │
│  deconvolved tiles → register → registered tiles → fuse → mosaic        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         Salvage Pipeline                                 │
│                                                                          │
│  1. Detect shifts:   target tiles vs reference tiles → coarse_shifts    │
│                                                                          │
│  2. Pre-stitch:      deconvolved tiles + shifted positions → mosaic     │
│                      (applies per-tile position adjustments)             │
│                                                                          │
│  3. Combine:         fused channels → Zarr for visualization            │
│                                                                          │
│  4. Slice:           shifted mosaic → tiles in reference coords         │
│                      (feeds back into standard pipeline)                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Detect Coarse Shifts

Compute per-tile shifts between the problematic round and a reference round.

### Command

```bash
preprocess register fix-shifts PATH IDX \
    --roi ROI \
    --rounds "ROUND_NAME" \
    --reference REFERENCE_ROUND \
    [--detect-only]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATH` | Workspace path |
| `IDX` | 0-based tile index to process |
| `--roi ROI` | Target ROI name |
| `--rounds "ROUND_NAME"` | Round(s) to process (e.g., `"1_9_17"`) |
| `--reference REFERENCE_ROUND` | Reference round for alignment |
| `--detect-only` | Only detect shifts, do not modify images |

### Inputs

Deconvolved images in `[ZC]YX` format with 2 fiducial frames appended:

```text
<workspace>/analysis/deconv/{round}--{roi}/{round}-{idx:04d}.tif
<workspace>/analysis/deconv/{reference}--{roi}/{reference}-{idx:04d}.tif
```

### Outputs

```text
<workspace>/analysis/deconv/shifts--{roi}/coarse_shifts.json
```

```json
{
  "reference": "2_10_18",
  "tiles": {
    "0001": { "1_9_17": {"dx": 150.0, "dy": -23.0} },
    "0002": { "1_9_17": {"dx": 148.0, "dy": -25.0} }
  }
}
```

### Shift Convention

- `dx > 0`: Target content shifted RIGHT relative to reference
- `dy > 0`: Target content shifted DOWN relative to reference
- To align: subtract shift from tile position (move position opposite to content drift)

---

## Step 2: Fuse with Coarse Shifts

Pre-stitch the problematic round with per-tile position adjustments.

### Command

```bash
preprocess stitch fuse PATH ROI \
    --round-name ROUND_NAME \
    [--downsample 1] \
    [--threads 8]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATH` | Workspace path |
| `ROI` | Target ROI name |
| `--round-name` | Round to fuse (must exist in coarse_shifts.json) |
| `--downsample` | Spatial downsampling (default: 1) |
| `--coarse-shifts` | Optional: override path to coarse_shifts.json |

**Auto-detection:** The coarse_shifts.json path is automatically detected at
`<workspace>/analysis/deconv/shifts--{roi}/coarse_shifts.json`. Fiducial frames
are also automatically extracted and fused.

### Image Format Handling

Deconvolved images are `[ZC]YX` — Z and C dimensions flattened into one axis.
The fuse command automatically:

1. **Removes fiducial frames** (2 frames from end)
2. **Reshapes to ZCYX** based on channel count from round name

For round `1_9_17` (3 underscore-separated bits = 3 channels):

```text
Input:   (Z*3 + 2, H, W)   [ZC]YX + 2 fiducials
Step 1:  (Z*3, H, W)       Remove fiducials
Step 2:  (Z, 3, H, W)      Reshape to ZCYX
```

### Per-Tile Position Adjustment

Each tile's position in the tile configuration is adjusted:

```python
adjusted_x = original_x - dx / downsample
adjusted_y = original_y - dy / downsample
```

This shifts tile positions opposite to the detected drift, aligning the mosaic
to reference coordinates.

### Outputs

```text
<workspace>/analysis/deconv/stitch--{roi}--shifted-{round}/
├── 00/                 # Z=0
│   ├── 00/             # Channel 0
│   │   ├── 0001.tif
│   │   └── fused_00-1.tif
│   ├── 01/             # Channel 1
│   └── 02/             # Channel 2
├── 01/                 # Z=1
│   └── ...
└── fid/                # Fiducials (1 channel, 2 Z-planes)
    ├── 00/             # Fiducial Z=0
    │   ├── 0001.tif
    │   └── fused_00-1.tif
    └── 01/             # Fiducial Z=1
```

Fiducial frames are treated as a single channel with `n_fids` Z-planes (typically 2).
This structure allows the slice-back step to reconstruct tiles with fiducials appended.

---

## Step 3: Combine to Zarr

Combine fused channel mosaics into a single Zarr array for visualization.

### Command

```bash
preprocess stitch combine PATH ROI \
    --round-name ROUND_NAME \
    [--chunk-size 2048]
```

### Outputs

```text
<workspace>/analysis/deconv/stitch--{roi}--shifted-{round}/fused.zarr
```

Shape: `(Z, Y, X, C)`

---

## Step 4: Slice Back into Tiles

Extract tiles from the shifted mosaic in [ZC]YX format with fiducials appended.

### Command

```bash
preprocess stitch slice PATH ROI \
    --round-name ROUND_NAME \
    [--tile-size 2048] \
    [--overwrite]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATH` | Workspace path |
| `ROI` | Target ROI name |
| `--round-name` | Round name (must have completed steps 1-3) |
| `--tile-size` | Output tile size in pixels (default: 2048) |
| `--overwrite` | Overwrite existing output files |

### Coordinate Logic

ImageJ sets mosaic origin (0,0) at `min(x), min(y)` from TileConfiguration.
Since the mosaic was built with *shifted* positions, the origin is:

```python
mosaic_origin = (min(shifted_x), min(shifted_y))
             = (min(original_x - dx), min(original_y - dy))
```

To extract tiles that work at *original* positions in the standard pipeline:

```python
slice_x = original_x - mosaic_origin_x
slice_y = original_y - mosaic_origin_y
```

### Outputs

```text
<workspace>/analysis/deconv/{round}--{roi}--repaired/
├── {round}-0001.tif    # (Z*C + 2, 2048, 2048) [ZC]YX + fiducials
├── {round}-0002.tif
└── ...
```

Output tiles are in the same [ZC]YX format as the original deconvolved images,
with fiducials appended. They can feed directly into the standard registration
pipeline.

---

## Complete Example

```bash
# 1. Detect shifts for round 1_9_17 relative to reference 2_10_18
preprocess register fix-shifts /data/workspace/analysis/deconv 1 \
    --roi roi1 \
    --rounds "1_9_17" \
    --reference 2_10_18

# 2. Fuse the problematic round (auto-detects coarse_shifts.json, includes fiducials)
preprocess stitch fuse /data/workspace roi1 \
    --round-name 1_9_17 \
    --threads 8

# 3. Combine into Zarr for visualization
preprocess stitch combine /data/workspace roi1 \
    --round-name 1_9_17

# 4. Slice back into tiles for standard pipeline
preprocess stitch slice /data/workspace roi1 \
    --round-name 1_9_17

# Result: /data/workspace/analysis/deconv/1_9_17--roi1--repaired/
# Contains reference-aligned tiles that can feed into standard registration
```

---

## Troubleshooting

### "Cannot reshape [ZC]YX image: N frames not divisible by M channels"

The frame count after removing fiducials must be divisible by channel count.

- Verify fiducial count (default: 2)
- Check round name reflects actual channels (`1_9_17` = 3 channels)

### Large shifts (>500px)

Indicates severe drift or wrong reference. Consider:

- Verify reference round selection
- Check for stage issues during acquisition
- Use a temporally closer reference

### Mosaic misalignment after fusion

If tiles don't align properly in the fused mosaic:

- Verify shift signs are correct (may need inversion)
- Check that all tiles have shifts in the JSON
- Confirm reference round tile positions are accurate

---

## Technical Notes

### Why Crop-Based (Integer) Shifting?

Standard sub-pixel registration uses interpolation, which:

- Smooths pixel values
- Can degrade spot detection
- Is slow for large shifts

Crop/pad shifting preserves exact pixel values — ideal for large integer offsets
where sub-pixel precision is unnecessary.

### Output Compatibility

| Standard Pipeline | Salvage Pipeline |
|-------------------|------------------|
| `registered--{roi}+{codebook}/` | `{round}--{roi}/` (source) |
| `stitch--{roi}+{codebook}/` | `stitch--{roi}--shifted-{round}/` |

The shifted outputs are suitable for visualization and QC. Full pipeline
compatibility (feeding back into spot calling) requires the slice step.

### Why Per-Tile Shifts?

While drift is often globally consistent, per-tile shifts handle:

- Slight tile-to-tile variations
- Non-uniform stage drift
- Edge effects near mosaic boundaries

Computing shifts per-tile and averaging would give a global estimate, but
applying per-tile allows finer correction.

---

## API Reference

### Image Format: [ZC]YX

Deconvolved images use `[ZC]YX` format where Z and C are flattened:

| Property | Value |
|----------|-------|
| Shape | `(Z*C + n_fids, H, W)` |
| Frame order | Z0C0, Z0C1, Z0C2, Z1C0, Z1C1, Z1C2, ... |
| Fiducials | Last `n_fids` frames (default: 2) |
| Channel count | Derived from round name: `"1_9_17"` → 3 channels |

### Path Contracts

| Resource | Path Pattern |
|----------|-------------|
| Deconvolved source | `<ws>/analysis/deconv/{round}--{roi}/` |
| Coarse shifts | `<ws>/analysis/deconv/shifts--{roi}/coarse_shifts.json` |
| Shifted output | `<ws>/analysis/deconv/stitch--{roi}--shifted-{round}/` |
| Fused zarr | `<ws>/analysis/deconv/stitch--{roi}--shifted-{round}/fused.zarr` |
| Shifted TileConfiguration | `<ws>/analysis/deconv/stitch--{roi}--shifted-{round}/TileConfiguration.shifted.txt` |
| Fiducial mosaics | `<ws>/analysis/deconv/stitch--{roi}--shifted-{round}/fid/{z:02d}/fused_00-1.tif` |
| Original TileConfiguration | `<ws>/stitch--{roi}/TileConfiguration.registered.txt` |
| Repaired output | `<ws>/analysis/deconv/{round}--{roi}--repaired/{round}-{idx:04d}.tif` |

### coarse_shifts.json Schema

```json
{
  "reference": "string",
  "tiles": {
    "<tile_idx>": {
      "<round_name>": {
        "dx": "<float: positive = content shifted RIGHT>",
        "dy": "<float: positive = content shifted DOWN>"
      }
    }
  }
}
```

### Output Directory Structure

```text
stitch--{roi}--shifted-{round}/
├── TileConfiguration.shifted.txt  # Shifted positions (used by slice)
├── 00/                 # Z=0
│   ├── 00/             # Channel 0
│   │   ├── 0001.tif    # Extracted tile
│   │   └── fused_00-1.tif
│   ├── 01/             # Channel 1
│   └── 02/             # Channel 2
├── 01/                 # Z=1
│   └── ...
├── fid/                # Fiducials
│   ├── 00/             # Fiducial Z=0
│   └── 01/             # Fiducial Z=1
└── fused.zarr          # Combined output (Z, Y, X, C)
```

### Function Contracts

#### `extract()`

New parameters for [ZC]YX handling:

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_channels_reshape` | `int \| None` | If set, reshape [ZC]YX to ZCYX with this many channels |
| `n_fids` | `int` | Number of fiducial frames to remove from end before reshape |
| `include_fiducials` | `bool` | If True, save fiducials to `fid/` structure |

**Preconditions:**
- If `n_channels_reshape` set, input must be 3D
- `(frames - n_fids) % n_channels_reshape == 0`

**Postconditions:**
- Main channels: `{out_path}/{z:02d}/{c:02d}/{tile}.tif`
- Fiducials: `{out_path}/fid/{fid_z:02d}/{tile}.tif`

#### `fuse()`

New parameters for round-name mode:

| Parameter | Type | Description |
|-----------|------|-------------|
| `coarse_shifts` | `Path \| None` | Override path to coarse_shifts.json |
| `round_name` | `str \| None` | Round name for deconvolved fusion |

**Modes:**
- `--codebook`: Standard registered fusion (existing behavior)
- `--round-name`: Deconvolved fusion with coarse shifts

**Preconditions (round-name mode):**
- `coarse_shifts.json` exists at auto-detected path or provided path
- `round_name` exists in the JSON

**Behavior:**
- Channel count = `len(round_name.split("_"))`
- Fiducials always extracted (2 frames)
- Per-tile shifts applied: `tile_x -= dx/downsample`
- Saves `TileConfiguration.shifted.txt` to stitch folder (for slice step)

#### `combine()`

New parameter:

| Parameter | Type | Description |
|-----------|------|-------------|
| `round_name` | `str \| None` | Round name for shifted fusion folder |

**Modes:**
- `--codebook`: Read from standard stitch folder
- `--round-name`: Read from shifted folder

#### `slice_mosaic()`

Slice shifted mosaic back into tiles in [ZC]YX format with fiducials.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `Path` | Workspace path |
| `roi` | `str` | Target ROI name |
| `round_name` | `str` | Round name (required) |
| `tile_size` | `int` | Output tile size in pixels (default: 2048) |
| `overwrite` | `bool` | Overwrite existing files |

**Preconditions:**
- `fused.zarr` exists (run `combine` first)
- `TileConfiguration.shifted.txt` exists in stitch folder (created by `fuse`)
- `TileConfiguration.registered.txt` exists (original positions)

**Coordinate Logic:**
```python
# Mosaic origin = min(shifted positions) from TileConfiguration.shifted.txt
origin_x = shifted_tc.df["x"].min()
origin_y = shifted_tc.df["y"].min()

# Slice at original position relative to shifted origin
slice_x = original_x - origin_x
slice_y = original_y - origin_y
```

**Postconditions:**
- Output: `{round}--{roi}--repaired/{round}-{idx:04d}.tif`
- Format: `(Z*C + n_fids, tile_size, tile_size)` matching original [ZC]YX
