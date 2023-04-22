# %%
from pathlib import Path

import click
import glymur
import glymur.jp2box
import numpy as np
from lxml.etree import Element, ElementTree


def dict_to_xml(tag: str, d: dict[str, str]):
    elem = Element(tag)
    for key, val in d.items():
        child = Element(key)
        child.text = str(val)
        elem.append(child)
    return ElementTree(elem)


def gen_inf(*, position: str, exposure_times: str, n_frames: int):
    return f"""binning = (1 x 1)
data type = 16 bit integers (binary, little endian)
frame dimensions = 2048 x 2048
number of frames = {n_frames}
x_start = 1
x_end = 2048
y_start = 1
y_end = 2048
position = {position}
exposure_times = {exposure_times}"""


@click.command()
@click.argument("path", type=click.Path(exists=True))
def main(path: Path):
    path = Path(path)
    jp2 = glymur.Jp2k(path)
    img = np.moveaxis(jp2[:], 2, 0)
    img.tofile(path.with_suffix(".dax"))
    metadata = dict([(x.tag, x.text) for x in jp2.box[-1].xml.getroot()])
    Path(path.with_suffix(".inf")).write_text(
        gen_inf(n_frames=img.shape[0], **metadata)
    )


if __name__ == "__main__":
    main()
