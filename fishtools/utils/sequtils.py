import random
import re
from typing import Any, Literal

import colorama
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.axes import Axes

table = str.maketrans("ACGTacgt ", "TGCAtgca ")
name_splitter = re.compile(r"(.+)_(.+):(\d+)-(\d+)")

c = re.compile(r"(\d+)S(\d+)M")
c2 = re.compile(r"(\d+)M")


def parse_cigar(s: str, m_only: bool = False) -> tuple[int, ...]:
    if not m_only:
        try:
            return tuple(map(int, c.findall(s)[0]))
        except IndexError:
            try:
                return 0, max(map(int, (c2.findall(s))))
            except ValueError:
                return 0, 0
            except IndexError:
                return 0, 0
    return tuple(map(int, c2.findall(s)))


def printc(seq: str):
    for c in seq:
        if c == "A" or c == "a":
            print(colorama.Fore.GREEN + c, end="")
        elif c == "T" or c == "t":
            print(colorama.Fore.RED + c, end="")
        elif c == "C" or c == "c":
            print(colorama.Fore.BLUE + c, end="")
        elif c == "G" or c == "g":
            print(colorama.Fore.YELLOW + c, end="")
        else:
            print(colorama.Fore.WHITE + c, end="")
    print(colorama.Fore.RESET)


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    # https://bioinformatics.stackexchange.com/a/3585
    return seq.translate(table)[::-1]


def pcr(seq: str, primer: str, primer_rc: str) -> str:
    loc = seq.find(primer)
    if loc == -1:
        raise ValueError(f"Primer {primer} not found in sequence {seq}")
    loc_rc = reverse_complement(seq[loc:]).find(primer_rc)
    if loc_rc == -1:
        raise ValueError(f"Primer {primer_rc} not found in sequence {seq}")
    return seq[loc : None if loc_rc == 0 else -loc_rc]


def gen_random_base(n: int) -> str:
    """Generate a random DNA sequence of length n."""
    return "".join(random.choices("ACGT", k=n))


def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


def equal_distance(total: int, choose: int) -> npt.NDArray[np.int_]:
    return np.linspace(0, total - 1, choose).astype(np.int_)


def plot_gc_content(ax: Axes, seq: str, window_size: int = 50, **kwargs: Any):
    """Plot windowed GC content on a designated Matplotlib ax."""
    if len(seq) < window_size:
        raise ValueError("Sequence shorter than window size")
    out = np.zeros(len(seq) - window_size + 1, dtype=float)
    curr = gc_content(seq[:window_size]) * window_size
    out[0] = curr
    for i in range(1, len(seq) - window_size):
        curr += (seq[i + window_size - 1] in "GC") - (seq[i - 1] in "GC")
        out[i] = curr
    out /= window_size

    ax.fill_between(np.arange(len(seq) - window_size + 1), out, **(dict(alpha=0.3) | kwargs))  # type: ignore
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel("GC (%)")


def stripplot(**kwargs: Any) -> Axes:
    import pandas as pd
    import seaborn as sns

    sns.set()

    df = pd.concat(pd.DataFrame({"x": v, "y": k}) for k, v in kwargs.items())
    return sns.stripplot(data=df, x="x", y="y", **(dict(orient="h", alpha=0.6)))  # type: ignore


def gen_plate(name: str, seqs: list[str], order: Literal["C", "F"] = "C") -> pl.DataFrame:
    if order == "C":
        wells = [f"{row}{col:02d}" for row in "ABCDEFGH" for col in range(1, 13)]
    else:
        wells = [f"{row}{col:02d}" for col in range(1, 13) for row in "ABCDEFGH"]

    return pl.DataFrame(
        {
            "Well Position": wells[: len(seqs)],
            "Name": name,
            "Sequence": seqs,
        }
    )
