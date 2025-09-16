# %%
from itertools import cycle
from typing import Callable, Collection, Iterable, Literal, Sequence, TypeVar, cast, overload

import numpy as np
import polars as pl
from expression.collections import Seq
from fishtools.mkprobes.screen_utils._alignment import gen_bowtie_index, gen_fasta
from fishtools.mkprobes.screen_utils.samframe import SAMFrame
from Levenshtein import distance
from loguru import logger

from fishtools.mkprobes.constants import DT, SP6
from fishtools.mkprobes.definitions import Filter
from fishtools.mkprobes.utils.seqcalc import hp, tm
from fishtools.mkprobes.utils.sequtils import gc_content, reverse_complement
from fishtools.utils.utils import TAny, slide

T = TypeVar("T", list[str], dict[str, str], dict[int, str])


def remove_chosen(lis: list[TAny], to_exclude: Iterable[int]) -> list[TAny]:
    return [x for i, x in enumerate(lis) if i not in to_exclude]


def extract_ok(generated: T, ok_keys: list[int] | list[str]) -> T:
    if isinstance(generated, dict):
        return {k: generated[k] for k in ok_keys}  # type: ignore
    return [generated[i] for i in cast(list[int], ok_keys)]


@overload
def run_filter(f: Filter, seqs: T, return_ok_seqs: Literal[False] = ...) -> pl.DataFrame: ...


@overload
def run_filter(f: Filter, seqs: T, return_ok_seqs: Literal[True]) -> tuple[pl.DataFrame, T]: ...


def run_filter(f: Filter, seqs: T, return_ok_seqs: bool = False) -> pl.DataFrame | tuple[pl.DataFrame, T]:
    idx_mapping = {s: i for i, s in enumerate(seqs)}
    if isinstance(seqs, dict):
        fasta = gen_fasta(seqs.values(), names=map(str, seqs.keys())).getvalue()
        df = f(fasta).with_columns(idx=pl.col("name").map_elements(idx_mapping.get))
    else:
        fasta = gen_fasta(seqs).getvalue()
        df = f(fasta).with_columns(idx=pl.col("name").cast(pl.UInt32))

    if return_ok_seqs:
        return df, extract_ok(seqs, df["idx"].to_list() if isinstance(seqs, list) else df["name"].to_list())
    return df


class Filters:
    @staticmethod
    def check_humouse(fasta: str, match_consec_thresh: int = 16):
        df = SAMFrame.from_bowtie(
            fasta,
            "data/humouse/humouse",
            seed_length=10,
            threshold=13,
            fasta=True,
        )

        return (
            df.group_by("name")
            .agg(pl.col("match_consec").max())
            .filter(pl.col("match_consec").lt(match_consec_thresh | pl.col("match_consec").is_null()))
        )

    @staticmethod
    def check_other_probes(fasta: str, match_consec_thresh: int = 11, match_thresh: int = 15):
        """Checks for homology to other probes.

        Need to use negative selection because there is a chance that a probe may only bind to itself and no one else,
        resulting in a false negative."""

        gen_bowtie_index(fasta, "temp", "amplifiers")
        bt = SAMFrame.from_bowtie(
            fasta, "temp/amplifiers", seed_length=9, threshold=12, fasta=True, no_reverse=True
        )

        inverted = (
            bt.filter(pl.col("name").ne(pl.col("transcript")))
            .group_by("name")
            .agg(match_consec=pl.col("match_consec").max(), match=pl.col("match").max())
            .filter(
                (pl.col("match_consec").lt(match_consec_thresh) & pl.col("match").lt(match_thresh)).is_not()
            )
        )

        return pl.DataFrame({"name": list(set(bt["name"].unique()) - set(inverted["name"].unique()))})

    @staticmethod
    def check_amplifiers(
        fasta: str,
        zero: dict[str, str],
        match_consec_thresh: int = 11,
        match_thresh: int = 15,
        mode: Literal["fw", "rc"] = "fw",
    ):
        """Check amplifiers for interactions with its own base.
        Expected amplifiers to be named i_j where i is the index of the base and j is the index of the candidate.

        Reverses the order of the base and candidates.
        bowtie2 seems to have issues when the "reads" are longer than the reference.
        The idea here then is to create a reference using the "reads" and then align the base to it.
        Then, filter out the reads that align exactly to the base (perfect match, aligning to itself).
        """

        gen_bowtie_index(fasta, "temp", "amplifiers")
        bt = (
            SAMFrame.from_bowtie(
                gen_fasta(zero.values(), names=zero.keys()).getvalue(),
                "temp/amplifiers",
                seed_length=9,
                threshold=11,
                fasta=True,
                no_reverse=mode == "fw",
                no_forward=mode == "rc",
            )
            .filter(pl.col("transcript").ne("*"))
            .with_columns(transcript="name", name="transcript")
        )

        inverted = (
            bt.filter(pl.col("name").str.extract(r"(\d+)_\d+").ne(pl.col("transcript")))
            .group_by("name")
            .agg(match_consec=pl.col("match_consec").max(), match=pl.col("match").max())
            .filter(
                (
                    pl.col("match_consec").lt(match_consec_thresh)
                    & pl.col("match").lt(match_thresh if match_thresh != -1 else 1000)
                ).is_not()
            )
        )
        return pl.DataFrame({"name": list(set(bt["name"].unique()) - set(inverted["name"].unique()))})


def gen_random_candidates(
    already_selected: list[str], seed: int = 4, min_dist: int = 6, n: int = 1000000
) -> list[str]:
    rand = np.random.default_rng(seed)
    samples = np.apply_along_axis("".join, 1, rand.choice(["A", "T", "C"], size=[n, 20], p=[0.25, 0.25, 0.5]))

    selected = [DT, *already_selected]
    generated: list[str] = []

    for i, seq in enumerate(samples):
        if i % 100000 == 0:
            logger.info(f"Iteration {i}. Total selected: {len(selected)}.")

        if "CCC" in seq or "AAAA" in seq or "TTTT" in seq:
            continue

        if not 0.4 <= gc_content(seq) <= 0.6:
            continue

        for cutted in slide(seq):
            if distance(cutted, selected[-1]) < min_dist:
                continue
            if not (52 <= tm(seq, "dna", formamide=0) <= 57):
                continue

            for s in selected:
                if distance(s, cutted) < min_dist:
                    break
            else:
                if hp(seq + "AA" + seq, "dna") <= 0:
                    selected.append(cutted)
                    generated.append(cutted)

    return generated


def screen_candidates(already_selected: list[str], seed: int = 4, min_dist: int = 6, n: int = 1000000):
    generated = gen_random_candidates(already_selected, seed, min_dist, n)
    _, filtered = run_filter(lambda fasta: Filters.check_humouse(fasta), generated, return_ok_seqs=True)
    seqs = already_selected + filtered

    okk = run_filter(lambda fasta: Filters.check_other_probes(fasta), seqs)
    return extract_ok(seqs, sorted(filter(lambda x: x >= len(already_selected) + 1, okk["idx"].to_list())))


def _gen_amplifier(base: str, seq: str, reps: int = 5, *, rc: bool = False) -> str:
    if rc:
        seq, base = map(reverse_complement, (seq, base))
    fillers = cycle(["AA", "AT", "TA", "TT"])
    out = ""
    for f, _ in zip(fillers, range(reps)):
        out += f" {f} {seq}"
    return base + out


def screen_amplifiers(
    base: Collection[str],
    candidates: Sequence[str],
    *,
    screens: list[str] | None = None,
    generator: Callable[[str, str], str],
    mode: Literal["fw", "rc"] = "fw",
    match_thresh: int = 15,
):
    """Screens for homology to human and mouse
    then screens for binding to `screens` (if provided) or `base`"""

    primaries: dict[str, str] = {
        f"{i}_{j}": generator(b, f) for i, b in enumerate(base) for j, f in enumerate(candidates)
    }

    _, passed = run_filter(lambda fasta: Filters.check_humouse(fasta), primaries, return_ok_seqs=True)
    _, passed = run_filter(
        lambda fasta: Filters.check_amplifiers(
            fasta,
            zero={str(i): b for i, b in enumerate(screens or base)},
            mode=mode,
            match_thresh=match_thresh,
        ),
        passed,
        return_ok_seqs=True,
    )

    pairing: dict[int, int] = {}
    chosen = set()
    for name in passed.keys():
        i, j = tuple(map(int, name.split("_")))
        if i not in pairing and j not in chosen:
            pairing[i] = j
            chosen.add(j)

    assert len(pairing) == len(base), (len(pairing), len(base))
    return pairing, [generator(z, candidates[pairing[i]]) for i, z in enumerate(base)]


# %%

to_gen = 36
PRIMARY_REV = "CAAACTAACCTCCTTCTTCCTCCTTCCA"
SECONDARY_REV = reverse_complement("TCACATCACACCTCTATCCATTATCAACCAC")

ref = pl.read_csv("data/readout_ref_with_amp.csv")["seq"].to_list()
zeroth_readouts = pl.read_csv("data/readout_ref.csv")

candidates = screen_candidates(ref, seed=59)
assert not set(candidates) & set(ref)

# %%
pairing, seqs = screen_amplifiers(
    zeroth_readouts["seq"].to_list()[:to_gen],
    candidates,
    screens=zeroth_readouts["seq"].to_list(),
    generator=lambda b, f: PRIMARY_REV + " " + _gen_amplifier(b, f) + " TTC",
)
# %%
pairing_2, seqs_2 = screen_amplifiers(
    base=Seq(range(to_gen)).map(lambda x: pairing.get(x, -1)).map(candidates.__getitem__).to_list(),
    candidates=remove_chosen(candidates, pairing.values()),
    screens=seqs,
    generator=lambda b, f: SECONDARY_REV + " " + _gen_amplifier(b, f, rc=True) + " TTC",
    mode="rc",
    match_thresh=-1,
)

# %%
splitter: Callable[[int], Callable[[str], str]] = lambda i, s=" ": lambda x: x.split(s)[i]

# sanity checks
assert not set(map(splitter(-2), seqs)) & set(map(splitter(-2), seqs_2))
assert list(map(splitter(-2), seqs)) == list(map(lambda x: reverse_complement(splitter(1)(x)), seqs_2))

assert not (set(map(splitter(-2), seqs)) | set(map(splitter(-2), seqs_2))) & set(ref)
assert list(map(splitter(1), seqs)) == zeroth_readouts["seq"].to_list()[:to_gen]
# %%
SP6_F = reverse_complement("ACGTGACTGCTCC" + SP6[:-1])

# %%
idt = pl.concat([
    pl.DataFrame({
        "Pool name": "Amp1-Aug23",
        "Sequence": list(map(lambda x: x.replace(" ", "") + SP6_F, seqs)),
    }),
    pl.DataFrame({
        "Pool name": "Amp2-Aug23",
        "Sequence": list(map(lambda x: x.replace(" ", "") + SP6_F, seqs_2)),
    }),
])

idt == pl.read_excel("amplifiers.xlsx")
# idt.write_excel("amplifiers2.xlsx")

# %%
assert idt.select(pl.col("Sequence").str.contains(reverse_complement(SP6)))["Sequence"].all()
# %%
import pyfastx

synthesized = list(map(lambda x: x.seq, pyfastx.Fasta("amplifiers.fasta")))

# %%
what = idt["Sequence"].to_list()
# %%
what[0]

# %%
final_readers = list(
    map(reverse_complement, (map(lambda x: x[-(len(SP6_F) + 3) - 20 : -(len(SP6_F) + 3)], synthesized)))
)[36:]
# %%
metas = {2: "TCCCAACACATCCTATCTCA", 3: "ACCCATTACTCCATTACCAT", 4: "TATCATCCTTACACCTCACT"}


# %%
def gen_idt(name: str, seq: str):
    return f"{name}\t{seq}\t\t25nm\tSTD"


for i in range(3):
    for idx in range(8 * i, 8 * (i + 1)):
        print(
            gen_idt(
                f"Linker{idx + 1:02d}-MetaC-{i + 2:02d}",
                final_readers[idx] + reverse_complement("C" + metas[i + 2]),
            )
        )

mod = lambda i: i % 3 + 2
for i in range(24, 30):
    print(
        gen_idt(
            f"Linker{i + 1:02d}-MetaC-{mod(i):02d}",
            final_readers[i] + reverse_complement("C" + metas[mod(i)]),
        )
    )
# %%
for k, v in {k: "/5AmMC6/C" + v + "/3AmMO/" for k, v in metas.items()}.items():
    print(gen_idt(f"MetaReadout-C-Amp{k:02d}", v))
# %%
