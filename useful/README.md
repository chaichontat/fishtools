# Input Requirements for MultiFISH Processing Workflow

## Overview

The multiplex FISH processing pipeline requires several input types to function correctly. This document outlines the required formats and organization for successful processing through the deconvolution, registration, and spot detection stages.

## Raw Image Requirements

### File Organization

Raw images should be organized as follows:

- Base workspace directory
- `[round_name]--[ROI]/` (e.g., `1_9_17--cortex/`)
  - `[round_name]-[index].tif` (e.g., `1_9_17-0001.tif`)

### Image Format

- **Format**: Multi-page TIFF files
- **Structure**:
- Last 1-2 frames: Fiducial marker images (typically Alexa Fluor 405)
- Earlier frames: Bit channel images arranged sequentially
- **Dimensions**: Typically 2048×2048 pixels per frame
- **Bit depth**: 16-bit (uint16)

### Metadata Requirements

Each TIFF file should include metadata with:

- `waveform`: JSON string or dictionary containing:
- Channel information (e.g., "ilm405", "ilm560", "ilm650", "ilm750")
- Laser power settings
- Sequence information

Example:

```json
{
 "ilm405": {"power": 0.4, "sequence": [0,0,0,1,1]},
 "ilm560": {"power": 0.3, "sequence": [1,0,0,0,0]},
 "ilm650": {"power": 0.5, "sequence": [0,1,0,0,0]},
 "ilm750": {"power": 0.6, "sequence": [0,0,1,0,0]}
}
```

### Codebook Requirements

Codebooks define the bit pattern for each gene target:

- Format: JSON file
- Structure: Dictionary mapping gene names to bit indices
- File location: Base workspace directory

Example:

```json
{
  "Gene1": [1, 9, 17],
  "Gene2": [2, 10, 18],
  "Blank1": [3, 11, 19]
}
```

### Calibration Requirements

Point Spread Function (PSF)

- Format: TIFF file with 3D PSF measurements
- File location: Referenced in DATA path ("PSF GL.tif")
- Structure: Z-stack of PSF images

### Chromatic Aberration Correction

- Format: Text files with affine transformation matrices
- File location: Referenced in DATA path (e.g., "560to650.txt")
- Content: 6 values representing 2×2 matrix and translation vector

Example:

```
0.998 0.001 -0.001 0.997 1.32 -0.45
```

### BaSiC Templates

- Format: Pickle (.pkl) files containing BaSiC correction objects
- Structure: One file per wavelength/channel
- File location: [workspace]/basic/[round_name]-[wavelength].pkl

### Expected Directory Structure

Before processing:

```
workspace/
├── 1_9_17--cortex/
│   ├── 1_9_17-0001.tif
│   ├── 1_9_17-0002.tif
│   └── ...
├── 4_12_20--cortex/
│   ├── 4_12_20-0001.tif
│   └── ...
├── basic/
│   ├── 1_9_17-405.pkl
│   ├── 1_9_17-560.pkl
│   └── ...
└── codebook.json
```

After processing:

```
workspace/
├── (raw data as above)
├── analysis/
│   └── deconv/
│       ├── 1_9_17--cortex/
│       └── ...
├── deconv_scaling/
│   ├── 1_9_17.txt
│   └── ...
├── fids--cortex/
│   └── ...
├── registered--cortex+codebook/
│   ├── _fids/
│   ├── reg-0001.tif
│   └── ...
├── shifts--cortex+codebook/
│   └── ...
└── opt_codebook/
    ├── global_scale.txt
    ├── percentiles.json
    └── ...
```

### Processing Order

- BaSiC correction generation (if not provided)
- Deconvolution (cli_deconv.py)
- Registration (cli_register.py)
- Channel optimization (align_batchoptimize.py)
- Spot detection (align_prod.py batch)

# Accelerated 3D Deconvolution Pipeline (cli_deconv.py)

## Overview and Purpose

The `cli_deconv.py` module implements a GPU-accelerated 3D deconvolution system for multiplexed fluorescence in situ hybridization (FISH) imaging data. Deconvolution improves the spatial resolution and signal-to-noise ratio of microscopy images by computationally removing out-of-focus blur. This implementation is specifically optimized for:

1. High-throughput processing of large image datasets
2. 3D-aware processing that accounts for the optical properties of the microscope
3. Consistent intensity scaling across images to enable accurate quantitative analysis

## Core Algorithm: Accelerated Lucy-Richardson Deconvolution

The primary algorithm is a modified Lucy-Richardson deconvolution based on Guo et al.'s "Accelerating iterative deconvolution and multiview fusion by orders of magnitude".

The entire deconvolution process runs on GPU using CuPy with the target time being similar to the actual imaging time.

## Multi-threaded Pipeline Architecture

To maximize throughput, the system uses a three-thread pipeline design:

1. **Reader Thread**: Loads images and applies BaSiC illumination correction
2. **Main Thread**: Performs GPU-accelerated deconvolution
3. **Writer Thread**: Handles file I/O in parallel with processing

This architecture ensures the GPU is continuously fed with data, eliminating I/O bottlenecks that would otherwise reduce GPU utilization.

## Illumination Correction

Before deconvolution, the system applies BaSiC (Background and Shading Correction) to correct for non-uniform illumination.

## Image Scaling System

The system implements a two-phase scaling approach:

### 1. Per-image Scaling

During deconvolution, each image channel is rescaled to maximize the 16-bit dynamic range for efficient storage. Without this rescaling, many deconvolved images would have very low intensity values and lose precision when stored as uint16. The scaling parameters (minimum value and scale factor) are stored in the image metadata.

### 2. Global Scaling Calculation

After all images are processed, global scaling parameters are calculated to ensure consistent intensity relationships across the dataset:

- **Global minimum**: High percentile (e.g., 99.9%) of all per-image minimum values
- **Global scale**: Low percentile (e.g., 0.1%) of all per-image scale factors

These global parameters represent boundary conditions that work across all images in the dataset.

### 3. Scaled Image Restoration

The global scaling approach enables downstream processing to restore consistent intensity scaling. The `scale_deconv` function in `cli_register.py` applies these parameters to reverse the 16-bit storage compression, ensuring that:

- The same intensity in the raw image maps to the same scaled value regardless of which field of view it appears in
- Signal intensities can be quantitatively compared across the entire dataset
- Measurement noise doesn't propagate through artificially high scaling factors

# Image Registration Pipeline in cli_register.py

## Key Components

### 1. Fiducial-Based Registration

Fiducial markers are fluorescent beads added to the sample that serve as reference points visible across all imaging channels. The registration process aligns these markers to compensate for spatial shifts:

- Fiducials are enhanced using Laplacian of Gaussian (LoG) filtering for improved detection
- The `align_fiducials` function in `fiducial.py` detects and matches corresponding spots between a reference channel and all other channels
- Spots are identified using `DAOStarFinder` from the astronomy package `photutils`
- Robust point matching via k-d trees and RANSAC ensures accurate alignment even with some spurious detections
- The algorithm automatically adjusts parameters (threshold, FWHM) when insufficient spots are detected

### 2. Chromatic Aberration Correction

Chromatic aberration causes different wavelengths of light to focus at slightly different positions, creating systematic distortions:

- The `Affine` class in `chromatic.py` applies pre-calibrated 3D affine transformations to each channel
- Shorter wavelengths (405nm, 488nm, 560nm) typically serve as reference channels
- Longer wavelengths (650nm, 750nm) are warped to align with the reference using transformation matrices
- Transformations are applied via SimpleITK's resampling filter with linear interpolation

### 3. Channel Processing Pipeline

For each image in the dataset:

1. **Image loading and parsing**:

- Images with multiple bit channels are loaded and separated
- Channel metadata (wavelength information, laser power) is extracted
- Fiducial markers are separated from experimental data

2. **Fiducial alignment**:

- Reference channel is selected (typically channel 560nm)
- Shifts are calculated between fiducials in all channels and the reference
- Prior shifts from previous registrations can be used to accelerate alignment

3. **Channel-specific processing**:

- Z-stacks are optionally collapsed using maximum projection
- Deconvolution scaling is applied (corrects for optical blurring)
- Optional BaSiC (Background and Shading Correction) for illumination correction
- Chromatic aberration correction via affine transformations
- Final shift correction based on fiducial alignment

4. **Output generation**:

- Aligned images are cropped to remove edge artifacts
- Optional downsampling for reduced file size
- Data is saved as multi-channel TIFF with comprehensive metadata

## Technical Details

### Fiducial Detection Algorithm

The spot detection uses an adaptive approach:

- Initial thresholds based on sigma-clipped statistics (number of standard deviations above background)
- FWHM (Full Width at Half Maximum) parameter controls expected spot size
- When spot detection fails, parameters are automatically adjusted:
- For too few spots: threshold is lowered or FWHM increased
- For too many spots: threshold is raised

### Chromatic Aberration Calibration

Transformation matrices are pre-calculated from calibration data:

- Beads imaged at different wavelengths are used to calculate systematic distortions
- Values are stored as 3×3 matrices (A) and translation vectors (t)
- The entire process is 3D-aware, maintaining proper z-axis alignment

### ROI-Based Processing

The ROI (Region of Interest) parameter allows:

- Processing specific anatomical regions separately
- Independent alignment within each region
- Maintaining separate calibrations for areas with different optical properties

### Additional Features

1. **Spillover correction**: Compensates for spectral bleed-through between adjacent channels
2. **Discards handling**: Allows selective exclusion of problematic channels
3. **Batch processing**: Efficiently processes multiple images in parallel
4. **Global deconvolution scaling**: Applies pre-calibrated intensity corrections

# Optimization Algorithm in align_prod and align_batchoptimize

## Overview

This algorithm implements an iterative channel balancing procedure for multiplexed FISH image analysis. It normalizes channel-to-channel intensity variations critical for accurate spot detection and decoding in experiments where signal strengths vary systematically across probes and fluorophores.

## Iterative Process

The `align_batchoptimize.py` script orchestrates multiple optimization rounds by calling `align_prod.py` functions:

1. **Spot detection and deviation calculation** - `align_prod.py optimize`
2. **Scale factor aggregation and update** - `align_prod.py combine`
3. **Threshold estimation** - `align_prod.py find-threshold`

## Technical Details

### Round 0: Initialization

- Calculates initial scaling factors using the 5th percentile of intensity maxima
- Establishes baseline normalization preventing dominance by exceptionally bright channels

### Rounds 1+: Optimization Loop

Each subsequent round performs:

1. **Channel Deviation Calculation**
   - Samples random subset of images
   - Applies current scaling factors
   - Detects spots using predefined parameters
   - Calculates channel-specific deviations from positive spots
   - Stores deviations with spot counts

2. **Scale Factor Update**
   - Aggregates deviations across images, weighted by spot count
   - Normalizes the deviation vector by its mean
   - Updates global scale factors
   - Quantifies convergence via coefficient of variation

3. **Threshold Refinement**
   - Computes percentile-based thresholds on high-pass filtered images
   - Calculates thresholds on the L2-norm across all channels
   - Stores thresholds for subsequent processing

### Key Components

#### Gaussian High-Pass Filtering

- Removes background while preserving spot features using adaptive sigma values

#### Zero-by-Channel-Magnitude Filtering

- Eliminates pixels with L2-norm below the threshold computed across all channels
- Maintains consistent signal detection regardless of contributing channels

#### Spot Detection and Decoding

- Aggregates adjacent bright pixels into spots
- Assigns spots to probable codewords using euclidean distance in feature space
- Uses parameters for distance threshold, intensity, and spot size requirements

## Strategy

The approach balances channel intensities by:

1. Identifying true positive spots using current scaling factors
2. Measuring deviation between observed and expected channel intensities
3. Adjusting scaling factors to compensate for systematic deviations
4. Computing cross-channel norm-based thresholds
5. Tracking convergence via coefficient of variation

This effectively compensates for variations in signal efficiency, photobleaching, and optical properties inherent in multiplexed FISH.

## Implementation

The algorithm operates on data subsets for computational efficiency, employing thread pooling for parallel processing. Scaling factors and thresholds are persistently stored and updated incrementally, allowing the process to be interrupted and resumed as needed.
