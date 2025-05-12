# %%
import json
import re
from itertools import chain
from pathlib import Path

import numpy as np
import rich_click as click
from loguru import logger

from fishtools.mkprobes.codebook.codebook import CodebookPicker, bit_count, n_to_bit

print(Path(__file__))
static = Path(__file__).resolve().parent.parent / "static"


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

        if np.any(bit_count(out[:cnt] ^ i) < min_dist):
            continue

        out[cnt] = i
        if ((np.array([out[cnt]])[:, None] & (1 << np.arange(n))) > 0).astype(int).sum() != on:
            raise ValueError(i)
        cnt += 1

    return out[:cnt]


def _generate(path: Path, n: int):
    x = _gen_mhd(n, 3, seed=0, min_dist=2)
    np.savetxt(path, out := n_to_bit(x, n, 3), fmt="%d", delimiter=",")
    return out


order = (
    list(chain.from_iterable([[i, i + 8, i + 16] for i in range(1, 9)])) + list(range(25, 50))
    # + list(range(31, 34))
)
paths = static.glob("*bit_on3_dist2.csv")
ns = {
    int(re.search(r"(\d+)", path.stem).group(1)): path.read_text().splitlines().__len__()
    for path in sorted(paths)
}
for n in range(10, 31):
    if n not in ns:
        ns[n] = len(_generate(static / f"{n}bit_on3_dist2.csv", n))


def gen_codebook(tss: list[str], offset: int = 0, n_bits: int | None = None, seed: int = 0):
    if n_bits is not None:
        n = int(n_bits)
        if (ns[n] - len(tss)) / ns[n] < 0.05:
            logger.warning("Less than 5% of coding capacity is blank")
    else:
        for bits, cnts in ns.items():
            if len(tss) < cnts:
                n = bits
                if (cnts - len(tss)) / cnts < 0.05:
                    logger.warning("Less than 5% of coding capacity is blank")
                break
        else:
            raise ValueError(f"No suitable codebook found. {len(tss)} genes found.")

    logger.info(f"Using {n}-bit codebook with capacity {ns[n]}.")

    cb = CodebookPicker(static / f"{n}bit_on3_dist2.csv", genes=tss)
    cb.gen_codebook(seed)
    c = cb.export_codebook(seed, offset=0)
    out = {k: sorted(order[x + offset] for x in v) for k, v in c.items()}

    # Remove bits that are perfect confounders of rounds
    FORBIDDEN = {
        *[(x, x + 8, x + 16) for x in range(1, 9)],
        *[(3 * i, 3 * i + 1, 3 * i + 2) for i in range(9, 13)],
    }

    to_swap = []
    for i, (k, v) in enumerate(out.items()):
        if tuple(v) in FORBIDDEN:
            to_swap.append(k)

    for i, k in enumerate(to_swap):
        out[f"Blank-{i + 1}"], out[k] = out[k], out[f"Blank-{i + 1}"]

    logger.info("Bits used: " + str(sorted(set(chain.from_iterable(out.values())))))

    # Sort by gene name
    return {k: out[k] for k in sorted(out, key=lambda x: (x.startswith("Blank"), x))}


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--offset", "-o", type=int, default=0)
@click.option(
    "--existing-codebook", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path)
)
@click.option("--n-bits")
def run(path: Path, offset: int = 0, existing_codebook: Path | None = None, n_bits: int | None = None):
    genes = path.read_text().splitlines()
    if offset and (existing_codebook is not None):
        raise ValueError("Must specify either offset or existing codebook")

    if existing_codebook:
        res = json.loads(existing_codebook.read_text())
        bits = set(chain.from_iterable(res.values()))
        offset = len(bits)
        logger.info(f"Using offset {offset} from existing codebook")

        if set(res) & set(genes):
            raise ValueError(f"Genes in existing codebook and input overlap: {set(res) & set(genes)}")

    generated = gen_codebook(genes, offset=offset, n_bits=n_bits)
    if existing_codebook and set(chain.from_iterable(generated.values())) & set(
        chain.from_iterable(res.values())
    ):
        raise Exception("Bits overlap")

    (path.with_suffix(".json")).write_text(json.dumps(generated, indent=2))


# %%
# %%
if __name__ == "__main__":
    run()
