# %%
import re
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from fishtools.mkprobes.codebook.codebook import CodebookPicker

print(Path(__file__))
static = Path(__file__).parent.parent.parent.parent / "static"


def _bit_count(arr: npt.NDArray[np.integer[Any]]) -> npt.ArrayLike:
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


def _gen_mhd(n: int, on: int, min_dist: int = 4, seed: int = 0):
    """
    Generate a Modified Hamming Distance (MHD) sequence.

    This function generates a sequence of `n`-bit numbers with a
    minimum Hamming distance of `min_dist` between each pair of numbers.
    The number of 1s in each number is equal to `on`.
    The sequence is generated using a random number generator with a specified `seed`.

    Parameters
    ----------
    n : int
        The number of bits in each number in the sequence.
    on : int
        The number of 1s in each number in the sequence.
    min_dist : int, optional
        The minimum Hamming distance between each pair of numbers in the sequence. Default is 4.
    seed : int, optional
        The seed for the random number generator. Default is 0.

    Returns
    -------
    np.ndarray
        A 1D numpy array containing the MHD sequence.

    Raises
    ------
    ValueError
        If the number of 1s in a generated number is not equal to `on`.

    Examples
    --------
    >>> print(gen_mhd(5, 2, 2, 0))
    array([ 9,  3,  5,  6, 10, 12, 17, 18, 20, 24], dtype=uint32)
    """

    assert n < 32
    rand = np.random.default_rng(seed)

    while (s := rand.integers(0, 2**n - 1)).bit_count() != on:
        ...

    out = np.zeros((2 ** (n - 1)), dtype=np.uint32)  # space saving. we're not getting more than half.
    out[0] = s
    cnt = 1

    for i in range(2**n):
        if not (i.bit_count() == on):
            continue

        if np.any(_bit_count(out[:cnt] ^ i) < min_dist):
            continue

        out[cnt] = i
        if (((np.array([out[cnt]])[:, None] & (1 << np.arange(n)))) > 0).astype(int).sum() != on:
            raise ValueError(i)
        cnt += 1

    return out[:cnt]


def _n_to_bit(arr: np.ndarray, n: int, on: int):
    """
    Convert an array of integers into a 2D array of binary representations.

    Each integer in the input array is represented as an `n`-bit binary number in the output array.
    The function also checks that the number of 1s in each binary representation is equal to `on`.

    Parameters
    ----------
    arr : np.ndarray
        A 1D numpy array of integers.
    n : int
        The number of bits to use for the binary representation of each integer.
    on : int
        The expected number of 1s in each binary representation.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row is the `n`-bit binary representation of the corresponding integer in the input array.

    Examples
    --------
    >>> import numpy as np
    >>> print(n_to_bit(np.array([1, 2, 4]), 3, 1))
    array([[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]])
    """

    if not isinstance(arr, np.ndarray):  # type: ignore
        raise TypeError("Input array must be a numpy array.")

    arr = (((arr[:, None] & (1 << np.arange(n)))) > 0).astype(int)
    if not np.all(arr.sum(axis=1) == on):
        raise ValueError(f"Number of 1s is not equal to {on=}.")
    return arr


def _generate(path: Path, n: int):
    x = _gen_mhd(n, 3, seed=0, min_dist=2)
    np.savetxt(path, out := _n_to_bit(x, n, 3), fmt="%d", delimiter=",")
    return out


order = (
    list(chain.from_iterable([[i, i + 8, i + 16] for i in range(1, 9)]))
    + list(range(25, 28))
    + list(range(31, 34))
)
paths = static.glob("*bit_on3_dist2.csv")
ns = {
    int(re.search(r"(\d+)", path.stem).group(1)): path.read_text().splitlines().__len__()
    for path in sorted(paths)
}
for n in range(10, 24):
    if n not in ns:
        ns[n] = len(_generate(static / f"{n}bit_on3_dist2.csv", n))


# @click.command()
# @click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
# @click.option("--offset", "-o", type=int, default=0)
def gen_codebook(tss: list[str], offset: int = 0):
    for bits, cnts in ns.items():
        if len(tss) < cnts:
            n = bits
            if (cnts - len(tss)) / cnts < 0.05:
                logger.warning("Less than 5% of coding capacity is blank")
            break
    else:
        raise ValueError(f"No suitable codebook found. {len(tss)} genes found.")

    logger.info(f"Using {n}-bit codebook with capacity {ns[n]}.")

    cb = CodebookPicker(f"../static/{n}bit_on3_dist2.csv", genes=tss)
    cb.gen_codebook(1)
    c = cb.export_codebook(1, offset=0)
    out = {k: list(map(lambda x: order[x + offset], v)) for k, v in c.items()}
    logger.info("Bits used: " + str(sorted(set(chain.from_iterable(out.values())))))
    return out


# %%
# %%
