# %%
from pathlib import Path

import numpy as np
import polars as pl

df = (
    pl.read_csv("neuroRef.csv")
    .with_columns(maxx=pl.max(pl.col("pc1.rot").abs(), pl.col("pc2.rot").abs()))
    .sort("maxx", descending=True)
)

# %%
import matplotlib.pyplot as plt

# %%
df[:140]
# %%
fpkm = pl.read_csv("../oligocheck/data/fpkm/combi99percentile.csv")
# %%
combi = df.join(fpkm, left_on="symbol", right_on="gene").sort("maxx", descending=True)

# %%
plt.hist(combi["value"], log=True, bins=100)
# %%
Path("starwork/tricycle.txt").write_text(
    "\n".join(combi.sort("value")[:-14].sort("maxx", descending=True)[:120]["symbol"])
)
# %%
f = dict(zip(fpkm["gene"], fpkm["value"]))
# %%
import json

gs = json.loads(Path("starwork2/synapse.json").read_text())
gs = ["-".join(g.split("-")[:-1]) for g in gs]

vs = {g: f.get(g, 0) for g in gs}
vs = {k: vs[k] for k in sorted(vs, key=vs.get)}
vs
# %%
df = pl.read_csv("synapse_max_tss.csv")
# %%
from fishtools import Dataset

ds = Dataset("data/mouse")
# %%
res = ds.ensembl.batch_convert(
    df["transcript_id"].apply(lambda x: x.split(".")[0]), "transcript_id", "transcript_name"
)
# %%
Path("starwork2/synapse.tss.txt").write_text("\n".join(sorted(r for r in res if r)))
# %%
