# https://gist.github.com/jwindhager/71fa5b149e85d61f83c9613e57d5b3f4

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import dilation, erosion


def label_erosion(labels: np.ndarray, footprint: np.ndarray | None = None) -> np.ndarray:
    grayscale_erosion = erosion(labels, footprint=footprint)
    grayscale_dilation = dilation(labels, footprint=footprint)
    return np.where(grayscale_erosion == grayscale_dilation, labels, 0)


def isotropic_label_erosion(
    labels: np.ndarray,
    radius: float,
    spacing: float | tuple[float, ...] | None = None,
) -> np.ndarray:
    nearest_bg_dists = distance_transform_edt(labels != 0, sampling=spacing)
    return np.where(nearest_bg_dists > radius, labels, 0)


def isotropic_label_dilation(
    labels: np.ndarray,
    radius: float,
    spacing: float | tuple[float, ...] | None = None,
) -> np.ndarray:
    nearest_fg_dists, nearest_fg_ind = distance_transform_edt(
        labels == 0, sampling=spacing, return_indices=True
    )
    nearest_labels = labels[tuple(nearest_fg_ind)]
    return np.where(nearest_fg_dists <= radius, nearest_labels, 0)


def isotropic_label_opening(
    labels: np.ndarray,
    radius: float,
    spacing: float | tuple[float, ...] | None = None,
) -> np.ndarray:
    eroded_labels = isotropic_label_erosion(labels, radius, spacing=spacing)
    return isotropic_label_dilation(eroded_labels, radius, spacing=spacing)


def isotropic_label_closing(
    labels: np.ndarray,
    radius: float,
    spacing: float | tuple[float, ...] | None = None,
) -> np.ndarray:
    dilated_labels = isotropic_label_dilation(labels, radius, spacing=spacing)
    return isotropic_label_erosion(dilated_labels, radius, spacing=spacing)
