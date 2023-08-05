# %%

from itertools import cycle
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import polars as pl
import pyfastx
from Levenshtein import distance

from fishtools.mkprobes.alignment import gen_bowtie_index, gen_fasta
from fishtools.utils.geneframe import GeneFrame
from fishtools.utils.seqcalc import hp, tm
from fishtools.utils.sequtils import gc_content, reverse_complement
from fishtools.utils.utils import slide

ref = pl.read_csv("data/readout_ref_with_amp.csv")


class Filter(Protocol):
    def __call__(self, fasta: str) -> pl.DataFrame:
        ...


# %%


def max_running_distance(a: str, b: str) -> int:
    maxcurr = 0
    curr = 0
    for x, y in zip(a, b):
        if x == y:
            curr += 1
        else:
            maxcurr = max(maxcurr, curr)
            curr = 0
    return max(maxcurr, curr)


dTmarker = "TTACACTCCATCCACTCAA"


fasta = pyfastx.Fasta("seq_generator/all.fasta")

rand = np.random.default_rng(4)
samples = np.apply_along_axis(
    "".join, 1, rand.choice(["A", "T", "C"], size=[1000000, 20], p=[0.25, 0.25, 0.5])
)


# %%
# toget = rand.integers(0, len(fasta), size=100000)
selected = [dTmarker, *ref["seq"]]
generated = []
dist = 6
for i, seq in enumerate(samples):
    if i % 100000 == 0:
        print(i, len(selected))

    if "CCC" in seq or "AAAA" in seq or "TTTT" in seq:
        continue

    if not 0.4 <= gc_content(seq) <= 0.6:
        continue

    for cutted in slide(seq):
        if distance(cutted, selected[-1]) < dist:
            continue
        if not (52 <= tm(seq, "dna", formamide=0) <= 57):
            continue

        # Distance check
        for s in selected:
            if distance(s, cutted) < dist:
                break
        else:
            if hp(seq + "AA" + seq, "dna") <= 0:
                selected.append(cutted)
                generated.append(cutted)


# %%
T = TypeVar("T", list[str], dict[str, str], dict[int, str])


def extract_ok(generated: T, ok_keys: list[int] | list[str]) -> T:
    if isinstance(generated, dict):
        return {k: generated[k] for k in ok_keys}
    return [generated[i] for i in cast(list[int], ok_keys)]

    # filtered = np.array(generated)[ok_idxs].tolist()
    # filtered_with_ref = list(ref) + filtered.tolist()
    # return filtered


# %%
@overload
def run_filter(f: Filter, seqs: T, return_ok_seqs: Literal[False] = ...) -> pl.DataFrame:
    ...


@overload
def run_filter(f: Filter, seqs: T, return_ok_seqs: Literal[True]) -> tuple[pl.DataFrame, T]:
    ...


def run_filter(f: Filter, seqs: T, return_ok_seqs: bool = False) -> pl.DataFrame | tuple[pl.DataFrame, T]:
    idx_mapping = {s: i for i, s in enumerate(seqs)}
    # key_mapping = {s: i for i, s in enumerate(seqs.keys())} if isinstance(seqs, dict) else idx_mapping

    fasta = (
        gen_fasta(seqs.values(), names=map(str, seqs.keys())).getvalue()
        if isinstance(seqs, dict)
        else gen_fasta(seqs).getvalue()
    )
    df = f(fasta).with_columns(
        # key=pl.col("name").apply(key_mapping.get),
        idx=pl.col("name").apply(idx_mapping.get),
    )
    if return_ok_seqs:
        return df, extract_ok(seqs, df["idx"].to_list() if isinstance(seqs, list) else df["name"].to_list())
    return df


# %%
def check_humouse(fasta: str, match_consec_thresh: int = 16):
    df = GeneFrame.from_bowtie(
        fasta,
        "data/humouse/humouse",
        seed_length=10,
        threshold=13,
        fasta=True,
    )

    return (
        df.groupby("name")
        .agg(pl.col("match_consec").max())
        .filter(pl.col("match_consec").lt(match_consec_thresh | pl.col("match_consec").is_null()))
    )


# %%
def gen_repeat(seq: str) -> str:
    return seq + "AA" + seq + "AT" + seq + "TA" + seq + "TT" + seq


def run2(fasta: str, base: list[str] | None = None, match_consec_thresh: int = 11, match_thresh: int = 15):
    gen_bowtie_index(
        fasta if base is None else gen_fasta(base, names=map(str, range(len(base)))).getvalue(),
        "temp",
        "amplifiers",
    )
    bt = GeneFrame.from_bowtie(
        fasta,
        "temp/amplifiers",
        seed_length=9,
        threshold=12,
        fasta=True,
    )

    return (
        bt.filter(pl.col("match_consec").lt(20))  # noself
        .groupby("name")
        .agg(match_consec=pl.col("match_consec").max(), match=pl.col("match").max())
        .filter(
            (pl.col("match_consec").lt(match_consec_thresh) & pl.col("match").lt(match_thresh))
            | pl.col("match").is_null()
        )
    )


# %%
ok, filtered = run_filter(lambda fasta: check_humouse(fasta), generated, return_ok_seqs=True)
# %%
# %%
seqs = list(ref["seq"]) + filtered
okk = run_filter(lambda fasta: run2(fasta), seqs)
final = extract_ok(seqs, sorted(filter(lambda x: x >= len(ref) + 1, okk["idx"].to_list())))
# %%
assert not set(final) & set(ref["seq"])
# round2 = run2(, base=filtered_with_ref)


# assert len(round2) == len(okk)
# final = np.array(filtered_with_ref)[idxs]
# %%

to_gen = 36
SP6 = "ATTTAGGTGACACTATAG"
sp6 = reverse_complement("ACGTGACTGCTCC" + SP6)
primary_rev = "CAAACTAACCTCCTTCTTCCTCCTTCCA"
secondary_rev = reverse_complement("TCACATCACACCTCTATCCATTATCAACCAC")


# %%


def gen_amp(base: str, seq: str, reps: int = 5, rc: bool = False) -> str:
    if rc:
        seq = reverse_complement(seq)
        base = reverse_complement(base)
    fillers = cycle(["AA", "AT", "TA", "TT"])
    out = ""
    for f, _ in zip(fillers, range(reps)):
        out += f"{seq} {f} "
    return base + out


zeroth_readouts = pl.read_csv("data/readout_ref.csv")
zeroth = zeroth_readouts["seq"].to_list()[:to_gen]
# primaries = final[:to_gen]
# secondaries = final[to_gen : to_gen * 2]


# primaries = [gen_amp(z, p) for z, p in zip(zeroth, primaries)]
# primaries = [primary_rev + " " + p + " " + sp6 for p in primaries]


# secondaries = [gen_amp(p, s, rc=True) for p, s in zip(primaries, secondaries)]
# secondaries = [secondary_rev + " " + s + " " + sp6 for s in secondaries]

# %%


# def run_round(f: Filter, )
#     passed_idxs = check_humouse(primaries)['name'].to_list()
#     return extract_ok(primaries, passed_idxs)


# %%


def run_primary(
    fasta: str, base: list[str] | None = None, match_consec_thresh: int = 11, match_thresh: int = 15
):
    gen_bowtie_index(
        fasta if base is None else gen_fasta(base, names=map(str, range(len(base)))).getvalue(),
        "temp",
        "amplifiers",
    )
    bt = GeneFrame.from_bowtie(
        fasta,
        "temp/amplifiers",
        seed_length=9,
        threshold=12,
        fasta=True,
    )

    bt.filter(pl.col("name").str.extract(r"(\d+)_\d+").eq(pl.col("transcript"))).unique("name")

    inverted = (
        bt.filter(pl.col("name").str.extract(r"(\d+)_\d+").ne(pl.col("transcript")))
        .groupby("name")
        .agg(match_consec=pl.col("match_consec").max(), match=pl.col("match").max())
        .filter((pl.col("match_consec").lt(match_consec_thresh) & pl.col("match").lt(match_thresh)).is_not())
    )
    return pl.DataFrame({"name": list(set(bt["name"].unique()) - set(inverted["name"].unique()))})


def gen_primary(base: list[str], candidates: list[str]):
    primaries: dict[str, str] = {}
    for i, b in enumerate(base):
        for j, f in enumerate(candidates):
            primaries[f"{i}_{j}"] = (primary_rev + gen_amp(b, f) + "C").replace(" ", "")

    _, passed = run_filter(lambda fasta: check_humouse(fasta), primaries, return_ok_seqs=True)
    breakpoint()
    _, passed = run_filter(lambda fasta: run_primary(fasta, base=base), passed, return_ok_seqs=True)

    pairing: dict[int, int] = {}
    chosen = set()
    for name in passed.keys():
        i, j = tuple(map(int, name.split("_")))
        if i not in pairing and j not in chosen:
            pairing[i] = j
            chosen.add(j)

    breakpoint()

    assert len(pairing) == len(base)
    return pairing, [gen_amp(z, candidates[pairing[i]]) for i, z in enumerate(base)]


gen_primary(zeroth, final)

# %%

secondaries = []
for i, z in enumerate(zeroth):
    for j in range(len(final)):
        idx = i * len(final) + j
        if idx in chosen:
            continue
        secondaries.append(gen_amp(z, final[idx]))

# filtered, filtered_with_ref = gen_filtered(ref["seq"], combi, ok["name"].to_list())

# %%
