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


def image_reader(filename, datatype, height, width, z):
    image_data = np.fromfile(filename, dtype=datatype, count=height * width * z)
    image_data = np.reshape(image_data, [z, width, height])
    return image_data


def parse_inf(f: str):
    return dict(x.split(" = ") for x in f.split("\n") if x)


@click.command()
@click.argument("path", type=click.Path(exists=True))
def main(path: Path):
    path = Path(path)
    print(path.with_suffix(".jp2"))

    inf = parse_inf(path.with_suffix(".inf").read_text())
    n_frames = int(inf["number of frames"])

    img = np.moveaxis(image_reader(path, np.uint16, 2048, 2048, n_frames), 0, 2)
    jp2 = glymur.Jp2k(path.with_suffix(".jp2"), data=img)
    jp2.append(
        glymur.jp2box.XMLBox(
            xml=dict_to_xml(
                "data",
                dict(position=inf["position"], exposure_times=inf["exposure_times"]),
            )
        )
    )


if __name__ == "__main__":
    main()
