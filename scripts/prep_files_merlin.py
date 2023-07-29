# %%
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from expression.collections import Seq

pl.Config.set_fmt_str_lengths(65)
# %% Codebook

n_bits = 24
bits_per_channel = 8
z_pos = str([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

superset = set(Seq(np.loadtxt("mhd4_24bit.csv", delimiter=",", dtype=int)).map(str))


def gen_inventory(name: str, readoutName: Iterable[str], bits: Iterable[int]):
    return {
        "name": name,
        "id": "",
        "barcodeType": "merfish",
        **dict(zip(readoutName, bits)),  # type: ignore
    }


cb = {k: v for k, v in sorted(json.loads(Path("codebook.json").read_text()).items(), key=lambda x: x[0])}
readoutName = [f"V00{i:02d}T8B1" for i in range(1, n_bits + 1)]

c = []


for k, v in cb.items():
    bits = Seq(range(1, n_bits + 1)).map(lambda x: 1 if x in v else 0)
    superset.remove(str(np.array(bits.to_list())))
    c.append(gen_inventory(k, readoutName, bits))

# Add blanks
for i, s in enumerate(superset, 1):
    bits = Seq(s[1:-1].split(" ")).map(int)
    gen_inventory(f"Blank-{i}", readoutName, bits)


cbout = pl.DataFrame(c)
cbout.write_csv("codebook.csv")

# %%
frame = {c: str(list(range(10 * (i - 1) + 1, 10 * i + 1))) for i, c in enumerate([560, 650, 750], 1)}
cmap = {0: 560, 1: 650, 2: 750}
color = Seq([0] * bits_per_channel + [1] * bits_per_channel + [2] * bits_per_channel).map(cmap.get)
bits = {i + 1: f"mod{i%8+1}" for i in range(n_bits)}
names = Seq.of(bits.values()).map(set).map(list).map(sorted).to_list()[0]

# %%
columns = [
    "channelName",
    "readoutName",
    "imageType",
    "imagingRound",
    "imageRegExp",
    "bitNumber",
    "color",
    "frame",
    "zPos",
    "fiducialImageType",
    "fiducialRegExp",
    "fiducialImagingRound",
    "fiducialFrame",
    "fiducialColor",
]
out = (
    pl.DataFrame(
        {
            "channelName": [f"bit{i}" for i in range(1, n_bits + 1)],
            "readoutName": readoutName,
            "imageType": Seq(range(1, n_bits + 1)).map(bits.get).map(str),
            "imagingRound": [i % 8 for i in range(n_bits)],
            "imageRegExp": r"(?P<imageType>[\w|_]+)-(?P<fov>[0-9]+)",
            "bitNumber": Seq(range(1, n_bits + 1)).map(str),
            "color": color,
            "frame": color.map(frame.get),  # type: ignore
            "zPos": z_pos,
            "fiducialFrame": 0,
            "fiducialColor": 405,
        }
    )
    .with_columns(
        fiducialImageType=pl.col("imageType"),
        fiducialRegExp=pl.col("imageRegExp"),
        fiducialImagingRound=pl.col("imagingRound"),
    )
    .select(columns)  # reorder columns
)

dapipolyA = [
    {
        "channelName": "PolyT",
        "readoutName": "",
        "imageType": "dapi_polyA",
        "imagingRound": -1,
        "imageRegExp": r"(?P<imageType>[\w|_]+)-(?P<fov>[0-9]+)",
        "bitNumber": "",
        "color": 488,
        "frame": "[11,12,13,14,15,16,17,18,19,20]",
        "zPos": z_pos,
        "fiducialImageType": "dapi_polyA",
        "fiducialRegExp": r"(?P<imageType>[\w|_]+)-(?P<fov>[0-9]+)",
        "fiducialImagingRound": -1,
        "fiducialFrame": 0,
        "fiducialColor": 405,
    },
    {
        "channelName": "DAPI",
        "readoutName": "",
        "imageType": "dapi_polyA",
        "imagingRound": -1,
        "imageRegExp": r"(?P<imageType>[\w|_]+)-(?P<fov>[0-9]+)",
        "bitNumber": "",
        "color": 405,
        "frame": "[1,2,3,4,5,6,7,8,9,10]",
        "zPos": z_pos,
        "fiducialImageType": "dapi_polyA",
        "fiducialRegExp": r"(?P<imageType>[\w|_]+)-(?P<fov>[0-9]+)",
        "fiducialImagingRound": -1,
        "fiducialFrame": 0,
        "fiducialColor": 405,
    },
]

pl.concat([out, pl.DataFrame(dapipolyA)]).write_csv("dataorganization.csv")

# %%

files = Seq(sorted(Path(".").glob("data/*.tif"))).map(lambda x: x.name)
pl.DataFrame(
    {
        "imageType": files.map(lambda x: x.split("-")[0]),
        "imagingRound": files.map(
            lambda x: str(int(m.group(1)) - 1) if (m := re.search(r"mod(\d)", x)) else "-1"
        ),
        "fov": files.map(lambda x: re.search(r"-(\d+)", x).group(1)).map(int),  # type: ignore
        "imagePath": files,
    }
).write_csv("filemap.csv")

# %%
