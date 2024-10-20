# %%
from pathlib import Path

import numpy as np
import rich_click as click
from tifffile import TiffFile, imwrite


def subtract(a: np.ndarray, b: np.ndarray, offset: int = 0):
    asub = a - offset
    bsub = b - offset
    return np.where(asub >= bsub, asub - bsub, 0)


def spillover_correction(spillee: np.ndarray, spiller: np.ndarray, corr: float):
    return subtract(spillee, spiller * corr, offset=0)


CHANNELS = {
    560: {"1", "2", "3", "4", "5", "6", "7", "8", "25"},
    650: {"9", "10", "12", "13", "14", "15", "16", "26"},
    750: {"17", "18", "19", "20", "21", "22", "23", "24", "27"},
}


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--median-scaling", type=float, default=1.5)
def run(path: Path, median_scaling: float = 1.5):
    with TiffFile(path) as tif:
        img = tif.asarray()
        meta = tif.shaped_metadata[0]  # type: ignore
        name_mapping = {b: i for i, b in enumerate(meta["key"])}

    nofid = img[0].astype(np.float32)
    avail_channels = {c: sorted(set(name_mapping) & CHANNELS[c], key=int) for c in CHANNELS}
    avail_idxs = {c: list(map(name_mapping.get, avail_channels[c])) for c in avail_channels}

    median = {name: np.median(nofid[idxs], axis=0).astype(np.uint16) for name, idxs in avail_idxs.items()}  # type: ignore

    perc = np.percentile(nofid, 20, axis=(1, 2))
    scale_factors = np.ones(nofid.shape[0], dtype=np.float32)
    for name, idxs in avail_idxs.items():
        currmin = np.min(perc[idxs])
        for i in idxs:
            scale_factors[i] = currmin / perc[i]
    nofid *= scale_factors[:, np.newaxis, np.newaxis]

    # Median subtraction
    for name, c in avail_idxs.items():
        nofid[np.s_[c]] = subtract(nofid[np.s_[c]], median[name] * median_scaling)  # type: ignore

    # Spillover correction
    spill_pairs = [(650, 560, 0.95)]
    for spiller, spillee, scale in spill_pairs:
        nofid[np.s_[avail_idxs[spillee]]] = spillover_correction(
            nofid[np.s_[avail_idxs[spillee]]],  # type: ignore
            nofid[np.s_[avail_idxs[spiller]]],  # type: ignore
            scale,
        )

    imwrite(
        path.with_name(path.stem + ".subtracted.tif"),
        nofid[np.newaxis, ...].astype(np.uint16),
        metadata=meta | {"subtracted": ["median", "spillover"]},
        compression=22610,
        compressionargs={"level": 0.75},
    )


if __name__ == "__main__":
    run()
