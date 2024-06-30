# %%
from pathlib import Path

import numpy as np
import rich_click as click
from loguru import logger
from tifffile import TiffFile

from fishtools.utils.pretty_print import progress_bar


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--perc_min", type=float, default=10, help="Percentile of the min")
@click.option("--perc_scale", type=float, default=5, help="Percentile of the scale")
def run(path: Path, perc_min: float = 10, perc_scale: float = 5):
    """Find the min and scale of the deconvolution for all files in a directory."""
    files = sorted(path.glob("*.tif"))
    n_c = len(path.resolve().stem.split("_"))
    n = len(files)

    deconv_min = np.zeros((n, n_c))
    deconv_scale = np.zeros((n, n_c))
    logger.info(f"Found {n} files")
    with progress_bar(len(files)) as pbar:
        for i, f in enumerate(files):
            try:
                with TiffFile(f) as tif:
                    deconv_min[i, :] = tif.shaped_metadata[0]["deconv_min"]
                    deconv_scale[i, :] = tif.shaped_metadata[0]["deconv_scale"]
            except Exception as e:
                logger.error(f"Error reading {f}: {e}")
            pbar()

    logger.info("Calculating percentiles")
    m_ = np.percentile(deconv_min, perc_min, axis=0)
    s_ = np.percentile(deconv_scale, perc_scale, axis=0)

    np.savetxt(path / "deconv_scaling.txt", np.vstack([m_, s_]))
    logger.info(f"Saved to {path / 'deconv_scaling.txt'}")


if __name__ == "__main__":
    run()
# %%

# %%
# with TiffFile(files[50]) as tif:
#     img = tif.pages[0].asarray()
#     curr_scale = np.array(tif.shaped_metadata[0]["deconv_scale"][0])
#     curr_min = np.array(tif.shaped_metadata[0]["deconv_min"][0])

#     print(tif.shaped_metadata[0]["deconv_min"])
#     print(
#         np.clip(
#             (img * (s_ / curr_scale)[0] + (s_ * (curr_min - m_))[0]),
#             0,
#             65535,
#         ).max()
#     )
# %%
