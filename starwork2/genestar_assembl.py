# %%

# %%
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import primer3
from Bio import Seq
from Bio.Restriction import BamHI, KpnI, RestrictionBatch

from fishtools import gen_fasta, hp, rc, tm
from fishtools.mkprobes.codebook.codebook import CodebookPicker
from fishtools.mkprobes.codebook.finalconstruct import assign_overlap
from fishtools.mkprobes.ext.external_data import Dataset
from fishtools.mkprobes.starmap.starmap import test_splint_padlock
from fishtools.mkprobes.utils.sequtils import reverse_complement


def check_primer(seq: str):
    assert primer3.calc_homodimer_tm(seq) < 45, seq
    assert primer3.calc_hairpin_tm(seq) < 45, (seq, primer3.calc_hairpin_tm(seq))


def make_header(seq: str):
    return seq + "GGTACC"


def make_footer(seq: str):
    return "GGATCC" + seq + "CCCTATAG"


hf = {
    "pad": (make_header("ACACGCACCTGTATC"), make_footer("CCGGCATTAT")),
    "splint": (make_header("CCCTCGTACTTCACTCA"), make_footer("GAGGCCACTATTAT")),
}

for p, s in hf.values():
    check_primer(p)
    check_primer(s)
    assert primer3.calc_heterodimer_tm(p, rc(s)) < 45
    assert 68 <= tm(p, "q5") <= 70, (tm(p, "q5"), "p", p)
    assert 68 <= tm(s, "q5") <= 70, (tm(s, "q5"), "s", s)
    print(p, rc(s))


# %%
tss = Path("starwork2/genestar.tss.txt").read_text().splitlines()

# %%


# ols = {}

# ols[ts] = assign_overlap("starwork/output", ts, restriction="_BamHIKpnI")
dfs = pl.concat(
    [
        pl.read_parquet(f"starwork2/output/{ts}_final_BamHIKpnI.parquet")
        # .sample(shuffle=True, seed=4, fraction=1)
        .sort(["priority", "hp"])[:30]
        for ts in tss
    ]
)
# %%

gene_to_ts = {re.split(r"-2\d\d", ts)[0]: ts for ts in tss}
notok = dfs.groupby("gene").agg(pl.count()).filter(pl.col("count") < 8)["gene"]
ok_ = dfs.groupby("gene").agg(pl.count()).filter(pl.col("count") >= 8)["gene"]

ok = [gene_to_ts[gene] for gene in ok_]
print(dfs.groupby("gene").agg(pl.count()).filter(pl.col("count") < 8))
# %%
# Path("starwork/tricycle.txt").write_text("\n".join(sorted(ok)))


def rotate(s: str, r: int):
    return s[r:] + s[:r]


def pad(s: str, target: int = 98):
    if len(s) > target + 2:
        raise ValueError("Too long")
    if len(s) > target:
        return s
    return s + "ACTCACCTAC"[: target - len(s)]


def backfill(seq: str, target: int = 137):
    return "TTCCACTAACTCACATGTCATGCATTTCTTCTTACC"[: max(0, target - len(seq))] + seq


def until_first_g(seq: str, target: str = "G"):
    r, target = rc(seq.upper()), target.upper()
    res, truncated = r[:6], r[6:]
    res += "" if res[-1] == target else truncated[: truncated.index(target) + 1]
    if len(res) > 12:
        raise ValueError("No G found")
    assert res[-1] == target
    return res


# assert (res["splint"].apply(rc).apply(lambda x: BsaI.search(Seq.Seq(x))).list.lengths() == 0).all()
# First must be C for KpnI.
# Last must be G for BamHI.
# %%


def generate_head_splint(padlock: str, rand: np.random.Generator):
    # Minimize the amount of secondary structures.
    # Must start with C.
    test = "TAA" + rc(padlock)
    # base_hp = hp(test, "dna")
    out = []
    for _ in range(10):
        cand = "CAC" + "".join(rand.choice(["A", "T", "C"], p=[0.125, 0.125, 0.75], size=3))
        if "CCCCC" in cand:
            continue
        out.append((cand, hp(cand + test, "dna")))

    cand, hp_ = min(out, key=lambda x: x[1])
    assert len(cand) == 6
    return cand


rand = np.random.default_rng(0)

res = dfs.with_columns(
    rotated=(
        # head
        pl.col("padlock").apply(lambda x: generate_head_splint(x, rand)).str.to_lowercase()
        + "ta"
        + pl.col("seq").apply(rc)
    ).apply(pad)
    + "G"
)
# %%
from itertools import cycle, islice

it = cycle("ATAAT")


def splint_pad(seq: str, target: int = 45):
    if len(seq) > target - 1:
        return "C" + seq
    return "C" + "".join(islice(it, target - len(seq) - 1)) + seq


res = (
    res.with_columns(
        splint_end=pl.col("rotated").apply(until_first_g).str.to_lowercase(),
        # mismatch
    )
    .with_columns(
        splint_=
        # + "TGTTGATGAGGTGTTGATGAT"
        # + "AA"
        pl.col("splint").apply(rc)
        + "TA"
        + pl.col("rotated").str.slice(0, 6).apply(rc)
        + pl.col("splint_end"),
    )
    .with_columns(splint_=pl.col("splint_").apply(splint_pad))
)
# %%

# uyu = res["splint_"].apply(lambda x: BsaI.catalyze(Seq.Seq(x)))
# uyu.filter(uyu.list.lengths() != 1)

assert (res["splint_"].apply(lambda x: BamHI.search(Seq.Seq(x))).list.lengths() == 0).all()
assert (res["splint_"].apply(lambda x: KpnI.search(Seq.Seq(x))).list.lengths() == 0).all()
assert (res["rotated"].apply(lambda x: BamHI.search(Seq.Seq(x))).list.lengths() == 0).all()
assert (res["rotated"].apply(lambda x: KpnI.search(Seq.Seq(x))).list.lengths() == 0).all()


def double_digest(s: str) -> str:
    return BamHI.catalyze(KpnI.catalyze(Seq.Seq(s))[1])[0].__str__()


# %%
pl.Config.set_fmt_str_lengths(100)

for s, r, ll in zip(res["splint_"], res["rotated"], res["splint_end"].str.lengths()):
    assert test_splint_padlock(s, r, lengths=(6, ll)), (s, r)

out = res.with_columns(
    # restriction scar already accounted for
    padlockcons=hf["pad"][0][:-1].lower() + pl.col("rotated") + hf["pad"][1][1:],
    splintcons=hf["splint"][0][:-1].lower() + pl.col("splint_") + hf["splint"][1][1:],
).with_columns(splintcons=pl.col("splintcons").apply(backfill))

for s, r, ll in zip(out["splintcons"], out["padlockcons"], out["splint_end"].str.lengths()):
    assert test_splint_padlock(*map(double_digest, (s, r)), lengths=(6, ll)), (s, r, ll)
# %%


# .filter(
#     pl.struct(["splintcons", "padlockcons"]).apply(
#         lambda x:
#     )
# )


assert (out["padlockcons"].str.lengths().is_between(139, 150)).all()
# assert (out["padlockcons"].str.lengths() == 150).all()


out.write_parquet("starwork/genestar.parquet")

# %%
Path("starwork/genestar_out.txt").write_text("\n".join([*out["padlockcons"], *out["splintcons"]]))
t7 = "TAATACGACTCACTATAGGG"
assert out["padlockcons"].str.contains(rc(t7)[:10]).all()

# %%
Path("starwork/genestarpad.fasta").write_text(gen_fasta(out["padlockcons"], names=range(len(out))).getvalue())
Path("starwork/genestarsplint.fasta").write_text(
    gen_fasta(out["splintcons"], names=range(len(out))).getvalue()
)

for name in ["genestarpad.fasta", "genestarsplint.fasta"]:
    subprocess.run(
        f'RepeatMasker -pa 64 -norna -s -no_is -species "mus musculus" starwork/{name}', shell=True
    )

# cons = dfs.with_columns(constructed=header + pl.col("seq") + footer)
# cons = cons.with_columns(constructed=pl.col("constructed").apply(backfill))
# %%
import pyfastx

out = []
for p, s in zip(
    pyfastx.Fastx("starwork/genestarpad.fasta.masked"), pyfastx.Fastx("starwork/genestarsplint.fasta.masked")
):
    if "N" not in s[1] and "N" not in p[1]:
        out.append(s[1])
        out.append(p[1])
print(len(out))
# %%

Path("starwork/genestar_out.txt").write_text("\n".join(out))

# %%
