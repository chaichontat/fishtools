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
from Bio.Restriction import BsaI, RestrictionBatch

from fishtools import T7, gen_fasta, rc, tm
from fishtools.mkprobes.codebook.codebook import CodebookPicker
from fishtools.mkprobes.codebook.finalconstruct import assign_overlap
from fishtools.mkprobes.ext.external_data import Dataset
from fishtools.mkprobes.starmap.starmap import test_splint_padlock
from fishtools.mkprobes.utils.sequtils import reverse_complement


def check_primer(seq: str):
    assert primer3.calc_homodimer_tm(seq) < 40
    assert primer3.calc_hairpin_tm(seq) < 40, primer3.calc_hairpin_tm(seq)


def make_header(seq: str, hang: str):
    assert len(hang) == 1
    return seq + "GGTCTC" + hang


def make_footer(seq: str):
    assert len(seq) == 5
    return seq + "GAGACC" + "ATTATCCCTATAGTGAG"


hf = {
    "pad": (make_header("GTGTGTGCGTTCA", "C"), make_footer("GCCAG")),
    "splint": (make_header("TGGAGCGTCTGTAT", "G"), make_footer("CGTCC")),
}

for p, s in hf.values():
    check_primer(p)
    check_primer(s)
    assert primer3.calc_heterodimer_tm(p, rc(s)) < 40
    assert 68 <= tm(p, "q5") <= 70, tm(p, "q5")
    assert 68 <= tm(s, "q5") <= 70, tm(s, "q5")
    print(p, rc(s))


# %%
tss = Path("starwork/tricycle.tss.txt").read_text().splitlines()

# %%

ols = {}
# for ts in tss:
# ols[ts] = assign_overlap("starwork/output", ts, restriction="_BsaI")
dfs = pl.concat(
    [
        pl.read_parquet(f"starwork/output/{ts}_final_BsaI.parquet")
        .sample(shuffle=True, seed=4, fraction=1)
        .sort("priority")[:14]
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
Path("starwork/tricycle.txt").write_text("\n".join(sorted(ok)))


def rotate(s: str, r: int):
    return s[r:] + s[:r]


def pad(s: str, target: int = 89):
    assert len(s) <= target
    return s + "ATAGTTATAT"[: target - len(s)]


def backfill(seq: str, target: int = 137):
    return "TTCCACTAACTCACATGTCATGCATTTCTTCTTACC"[: max(0, target - len(seq))] + seq


# assert (res["splint"].apply(rc).apply(lambda x: BsaI.search(Seq.Seq(x))).list.lengths() == 0).all()

res = dfs.with_columns(
    rotated=pl.col("seq").apply(rc).apply(pad).apply(lambda x: rotate(x, 20 - 6 - 3), return_dtype=pl.Utf8)
).with_columns(
    splint_="TGTTGATGAGGTGTTGATGAT"
    + "AA"
    + pl.col("splint").apply(rc)
    + "ATA"  # mismatch
    + pl.col("rotated").str.slice(0, 6).apply(rc)
    + pl.col("rotated").str.slice(-6, 6).apply(rc)
)


# uyu = res["splint_"].apply(lambda x: BsaI.catalyze(Seq.Seq(x)))
# uyu.filter(uyu.list.lengths() != 1)

assert (res["splint_"].apply(lambda x: BsaI.search(Seq.Seq(x))).list.lengths() == 0).all()

# %%
pl.Config.set_fmt_str_lengths(100)

for s, r in zip(res["splint_"], res["rotated"]):
    assert test_splint_padlock(s, r)

out = (
    res.with_columns(
        padlockcons=hf["pad"][0] + pl.col("rotated") + hf["pad"][1],
        splintcons=hf["splint"][0] + pl.col("splint_") + hf["splint"][1],
    )
    .filter(
        pl.struct(["splintcons", "padlockcons"]).apply(
            lambda x: test_splint_padlock(
                BsaI.catalyze(Seq.Seq(x["splintcons"]))[1].__str__(),
                BsaI.catalyze(Seq.Seq(x["padlockcons"]))[1].__str__(),
            )
        )
    )
    .with_columns(splintcons=pl.col("splintcons").apply(backfill))
)


assert (out["splintcons"].str.lengths() == 137).all()
assert (out["padlockcons"].str.lengths() == 137).all()


out.write_parquet("starwork/tricycle.parquet")

# %%


# %%
# cons = dfs.with_columns(constructed=header + pl.col("seq") + footer)
# cons = cons.with_columns(constructed=pl.col("constructed").apply(backfill))
# %%

t7 = "TAATACGACTCACTATAGGG"

assert out["padlockcons"].str.contains(rc(t7)[:10]).all()
# assert cons["constructed"].str.lengths().max() <= 150, cons["constructed"].str.lengths().max()

Path("starwork/tricyclepad.fasta").write_text(gen_fasta(out["padlockcons"], names=range(len(out))).getvalue())
Path("starwork/tricyclesplint.fasta").write_text(
    gen_fasta(out["splintcons"], names=range(len(out))).getvalue()
)

for name in ["tricyclepad.fasta", "tricyclesplint.fasta"]:
    subprocess.run(
        f'RepeatMasker -pa 64 -norna -s -no_is -species "mus musculus" starwork/{name}', shell=True
    )

# cons = dfs.with_columns(constructed=header + pl.col("seq") + footer)
# cons = cons.with_columns(constructed=pl.col("constructed").apply(backfill))
# %%
import pyfastx

out = []
for p, s in zip(
    pyfastx.Fastx("starwork/tricyclepad.fasta.masked"), pyfastx.Fastx("starwork/tricyclesplint.fasta")
):
    if "N" not in s[1] and "N" not in p[1]:
        out.append(s[1])
        out.append(p[1])
print(len(out))
# %%

Path("starwork/tricycle_out.txt").write_text("\n".join(out))

# %%
