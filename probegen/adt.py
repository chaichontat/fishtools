# %%
import polars as pl

from fishtools import rc
from fishtools.mkprobes.utils.sequtils import gen_idt

df = pl.read_csv("data/readout_ref_filtered.csv")
# %%
# [5Acryd]AATC[cnvK]TCAACACCTCATCAA[Ps]C[Ps]A[Ps][Inv-dT]
ANCHOR = "ATGTTGATGAGGTGTTGAT*G*A*/3invdT/"
SPACER = "AAT"


def gen(seq: str):
    return "/5AmMC6/" + rc(seq) + SPACER + ANCHOR


# %%
for i in range(51, 54):
    print(gen_idt(f"MER-ADT-{i}", gen(df.filter(pl.col("id") == i)[0]["seq"].item()), "100nm"))

# %%
