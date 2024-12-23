# %%
import re
from pathlib import Path

name = "/disk/chaichontat/2024/sv101_ACS/WGAfiducial_fiducial_tdT_29_polyT--noRNAse_big"
files = list(Path(name).glob("*.tif"))
for file in files:
    idx = file.name.split("-")[-1]
    outname = "WGAfiducial_fiducial_tdT_29_polyT-" + idx
    if outname in [x.name for x in Path(name).glob("*.tif")]:
        raise ValueError(f"File {outname} already exists.")
    file.rename(file.with_name(outname))


import json

# %%
from tifffile import TiffFile, imread, imwrite

files = list(Path(f"/fast2/jaredtest/{to}").glob("*.tif"))
for file in files:
    with TiffFile(file) as tif:
        if len(tif.pages) == 89:
            print(file)
            img = imread(file)
            fid = img[[-1]]
            nofid = img[:-1].reshape(22, 4, 2048, 2048)[:, [1, 2, 3], :, :].reshape(-1, 2048, 2048)
            nofid = np.concatenate([nofid, fid])
            try:
                m = tif.shaped_metadata[0]
            except IndexError:
                m = json.loads(tif.imagej_metadata["waveform"])

            m["ilm488"]["power"] = 0
            imwrite(file, nofid, metadata=m, imagej=True, compression=22610, compressionargs={"level": 0.65})

# %%
