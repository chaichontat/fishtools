# %%
import json
import re
from itertools import chain
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
from loguru import logger

from fishtools.mkprobes.codebook.codebook import CodebookPicker


def hamming(a: int, b: int):
    return (a ^ b).bit_count()


def bit_count(arr: npt.NDArray[np.integer[Any]]) -> npt.ArrayLike:
    # Make the values type-agnostic (as long as it's integers)
    t = arr.dtype.type
    mask = np.array(-1).astype(arr.dtype)
    s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
    s33 = t(0x3333333333333333 & mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & mask)
    s01 = t(0x0101010101010101 & mask)

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (arr.itemsize - 1))


def gen_mhd(n: int, on: int, min_dist: int = 4, seed: int = 0):
    assert n < 32
    rand = np.random.default_rng(seed)

    while (s := rand.integers(0, 2**n - 1)).bit_count() != on:
        ...

    # out = [s]
    out = np.zeros((2 ** (n - 1)), dtype=np.uint32)  # space saving. we're not getting more than half.
    out[0] = s
    cnt = 1

    for i in range(2**n):
        if not (i.bit_count() == on):  # or i.bit_count() == 5):
            continue

        if np.any(bit_count(out[:cnt] ^ i) < min_dist):
            continue

        # if any(hamming(i, j) < min_dist for j in out[:cnt]):
        #     continue

        out[cnt] = i
        if (((np.array([out[cnt]])[:, None] & (1 << np.arange(n)))) > 0).astype(int).sum() != on:
            raise ValueError(i)
        cnt += 1

    return out[:cnt]


def n_to_bit(arr, n: int, on: int):
    # n = int(np.max(arr)).bit_length()
    # assert 1 << (n + 1) > np.max(arr) >= 1 << (n)

    arr = (((arr[:, None] & (1 << np.arange(n)))) > 0).astype(int)
    print(arr.shape)
    assert np.all(arr.sum(axis=1) == on)
    return arr


def generate(n: int):
    x = gen_mhd(n, 3, seed=0, min_dist=2)
    np.savetxt(f"static/{n}bit_on3_dist2.csv", n_to_bit(x, n, 3), fmt="%d", delimiter=",")


# %%


order = list(chain.from_iterable([[i, i + 8, i + 16] for i in range(1, 9)])) + list(range(25, 33))
# %%

paths = Path("/home/chaichontat/fishtools/static").glob("*bit_on3_dist2.csv")
ns = {
    int(re.search(r"(\d+)", path.stem).group(1)): path.read_text().splitlines().__len__()
    for path in sorted(paths)
}


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--offset", "-o", type=int, default=0)
def gen_cb(path: Path, offset: int = 0):
    tss = path.read_text().splitlines()

    for bits, cnts in ns.items():
        if len(tss) < cnts:
            n = bits
            if (cnts - len(tss)) / cnts < 0.05:
                logger.warning("Less than 5% of coding capacity is blank")
            break
    else:
        raise ValueError(f"No suitable codebook found. {len(tss)} genes found.")

    logger.info(f"Using {n}-bit codebook with capacity {ns[n]}.")
    # mapping = {}
    # for i in range(n):
    #     d, m = divmod(i, 4)
    #     mapping[i + 1] = 8 * d + offset + m

    # GENE_PATH = Path(f"{mode}.txt")
    # tss = pl.concat([get_transcripts(ds, gene, mode="appris") for gene in GENE_PATH.read_text().splitlines()])
    # Path(f"{mode}.tss.txt").write_text(
    #     "\n".join(tss.groupby("gene_id").apply(parse)["transcript_name"].sort())
    # )

    cb = CodebookPicker(f"../static/{n}bit_on3_dist2.csv", genes=tss)
    cb.gen_codebook(1)
    c = cb.export_codebook(1, offset=0)
    Path(f"{path.stem.split('.')[0]}.json").write_text(
        json.dumps(out := {k: list(map(lambda x: order[x + offset], v)) for k, v in c.items()}, default=int)
    )

    print(set(chain.from_iterable(out.values())))


if __name__ == "__main__":
    gen_cb()

# %%
