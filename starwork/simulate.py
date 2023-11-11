# %%
from collections import deque
from pathlib import Path

import polars as pl
from Bio import Restriction, Seq

from fishtools import SAMFrame, gen_fasta, rc
from fishtools.mkprobes.starmap.starmap import test_splint_padlock

t7promoter = "TAATACGACTCACTATAGGG"
bits = pl.read_csv("data/readout_ref_filtered.csv")


def t7(seq: str):
    try:
        idx = seq.index(t7promoter)
        return seq[idx + len(t7promoter) - 3 :]  # First base is G.
    except ValueError:
        ...

    idx = rc(seq).index(t7promoter)
    return rc(seq)[idx + len(t7promoter) - 3 :]


def rt(seq: str, primer: str):
    try:
        res = rc(seq).index(primer)
    except ValueError:
        raise ValueError("Primer not found.")
    return rc(seq)[res:]


def digest(seq: str):
    return Restriction.BsaI.catalyze(Seq.Seq(seq))


# %%
from pathlib import Path

ss = Path("starwork/tricycle_out.txt").read_text().splitlines()


# %%
def process(seq: str, rt_primer: str):
    t7ed = t7(seq + "TCGTATTA")
    rted = rt(t7ed, rt_primer)
    digested = digest(rted)

    assert len(digested) == 3
    return str(digested[1])


# sp = process(ss[150], "AACTCGTACTTCACTCGGGTCTCA")
# pad = process(ss[151], "ACACGCACTGTATCGGTCTCC")
sp = process(ss[150], "TGGAGCGTCTGTATGGTCTCG")
pad = process(ss[151], "GTGTGTGCGTTCAGGTCTCC")

# %%
df = SAMFrame.from_bowtie(
    gen_fasta(
        [sp.split("TGTTGATGAGGTGTTGATGATAA")[1][:-15], pad[9:29]], names=["splint", "padlock"]
    ).getvalue(),
    "data/mouse/txome",
    seed_length=12,
    threshold=16,
    n_return=200,
    fasta=True,
    no_forward=True,
)

# %%
sps = set(df.filter(pl.col("name") == "splint").sort("match_consec")["transcript"])
pads = set(df.filter(pl.col("name") == "padlock").sort("match_consec")["transcript"])
filtered = df.filter(pl.col("transcript").is_in(sps & pads)).sort(["transcript", "name"])

idxs = (filtered[0, "pos"] + filtered[0, "match_consec"], filtered[1, "pos"])
assert 0 <= idxs[1] - idxs[0] < 3
assert test_splint_padlock(sp[-12:], pad)


# %%
def rotate(s: str, r: int):
    return s[r:] + s[:r]


(de := deque(pad)).rotate(-9)
rcaed = rc("".join(de))
assert len([bit for bit in bits["seq"] if rc(bit) in rcaed]) == 3

# %%
# %%


# %%
