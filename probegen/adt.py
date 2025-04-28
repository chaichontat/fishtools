# %%
import polars as pl

from fishtools import rc
from fishtools.mkprobes.utils.sequtils import gen_idt

df = pl.read_csv("data/readout_ref_filtered.csv")
# %%
# https://www.glenresearch.com/reports/gr30-21
# cnvK needs to be prefixed by A
# [5Acryd]AATCATCAA[cnvK]ACCTCATCAA[Ps]C[Ps]A[Ps][Inv-dT]
#           AGTAGTT G    TGGAGTAGTT    G    T    A

#          TGTTGATGAGGTGTTGATGAT
ANCHOR = "ATGTTGATGAGGTGTTGAT*G*A*/3AmMO/"
SPACER = "AAT"


def gen(seq: str):
    return rc(seq) + SPACER + ANCHOR


# %%
for i in range(40, 46):
    print(gen_idt(f"MER-ADT-{i}", gen(df.filter(pl.col("id") == i)[0]["seq"].item()), "100nm"))

# %%
for i in range(40, 46):
    print(
        gen_idt(
            f"Readout-dual-{i}",
            "/5AmMC6/" + df.filter(pl.col("id") == i)[0]["seq"].item() + "/3AmMO/",
            "100nm",
        )
    )

# %%
