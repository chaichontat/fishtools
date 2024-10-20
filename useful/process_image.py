# %%
import json
import re
from pathlib import Path

import numpy as np
import rich_click as click
import tifffile
from loguru import logger
from loopy.sample import Sample

# %%


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.argument("output", type=click.Path())
def main(path: Path, output: str):
    path = Path(path)
    prefix = "fused*"
    paths = sorted(path.rglob(prefix))
    logger.info("Fusing " + str([x.name for x in paths]))
    imgs = [tifffile.imread(x) for x in paths]
    res = np.stack(imgs)
    del imgs
    logger.info(f"Writting to {output}.")
    tifffile.imwrite(output, res, bigtiff=True, compression="zlib")


if __name__ == "__main__":
    main()
# %%

# img = np.clip(res // 256, a_min=0, a_max=255).astype(np.uint8)
# %%
# tifffile.imwrite("set0.tif", res, bigtiff=True)


# %%


import json
from pathlib import Path

fd = json.loads(Path("/fast2/fishtools/starwork3/hcrhuman.json").read_text())

chans = [x.split("-")[0] for x in sorted(fd, key=lambda x: fd[x][0])]

# %%
cb = json.loads(Path("/fast2/fishtools/starwork3/ordered/alina.json").read_text())
cb = {k: v[0] for k, v in cb.items()}
chans = [
    k
    for k in sorted(cb, key=lambda x: cb[x])
    # if not (k.startswith("Gng7") or k.startswith("Calm1") or k.startswith("mt") or k.startswith("Malat"))
]
chans.insert(-1, "16")

chans += ["all", "b560", "b650", "b750", "fiducial", "tdT", "polyA", "WGAfiducial"]
# chans += ["all", "b560", "b650", "b750", "polyA"]

chans = [re.sub(r"-\d\d\d", "", c) for c in chans]
chans = [re.sub(r"-", "", c) for c in chans]


# %%

# tifffile.imwrite("first100_8bitdown.tif", tifffile.imread("first100_8bit.tif")[:, ::2, ::2])
# From https://libd-spatial-dlpfc-loopy.s3.amazonaws.com/VisiumIF/sample.tif
tiff = Path("/disk/chaichontat/2024/sv101_ACS/registered--noRNAse_big/stitch/fused.tif")
c = chans
scale = 0.216e-6
quality = 100

Sample(
    name="sv101_ACS", path="/disk/chaichontat/2024/sv101_ACS/registered--noRNAse_big/stitch/samui/sv101_ACS"
).add_image(
    tiff,
    channels=chans,
    scale=scale,
    quality=quality,
    # defaultChannels={"blue": "Cx3cl1", "red": "Slc17a7", "green": "Plcxd2", "yellow": "Mal"},
).write()

# %%
# import rasterio

# with rasterio.open("/home/chaichontat/samui/scripts/alinastar2/first100_8bitdown_1.tif_", "r") as src:
#     print([src.overviews(i) for i in src.indexes])

# %%
