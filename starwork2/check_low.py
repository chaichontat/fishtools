# %%
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import primer3
from Bio import Seq

from fishtools.mkprobes.candidates import get_candidates
from fishtools.mkprobes.codebook.codebook import hash_codebook
from fishtools.mkprobes.ext.external_data import Dataset

# ds = Dataset("data/mouse")
# %%
path = Path("starwork2/synapse.json")
codebook = json.loads(path.read_text())
tss = list(codebook)
hsh = hash_codebook(codebook)

dfs = pl.concat(
    [
        pl.read_parquet(f"starwork2/output/{ts}_final_BamHIKpnI_{hsh}.parquet")
        # .sample(shuffle=True, seed=4, fraction=1)
        .sort(["priority", "hp"])[:30]
        for ts in tss
    ]
)

# %%

gene_to_ts = {"-".join(ts.split("-")[:-1]): ts for ts in tss}
ts_to_gene = {v: k for k, v in gene_to_ts.items()}
baddies = dfs.groupby("gene").agg(pl.count()).filter(pl.col("count") < 7).sort("count")
print(baddies)


# %%
try:
    allows = defaultdict(list, json.loads(path.with_suffix(".acceptable.json").read_text()))
except FileNotFoundError:
    allows = defaultdict(list)


for i in range(len(baddies)):
    gene = gene_to_ts[baddies[i, 0]]
    print(gene)
    while True:
        try:
            get_candidates(
                ds,
                gene=gene,
                output="starwork2/output/",
                ignore_revcomp=True,
                allow=allows.get(gene),
                realign=False,
            )
            off = pl.read_csv(f"starwork2/output/{gene}_offtarget_counts.csv")
        except Exception as e:
            ...
        resp = input(f"{gene}? ")
        if resp:
            allows[gene] += [x.strip() for x in resp.split(",")]
        else:
            break

print("done")

# %%


path.with_suffix(".acceptable.json").write_text(json.dumps(allows))

# %%

subprocess.run(
    f"python starwork2/cs.py data/human {path} --onlygene {','.join(allows)}",
    shell=True,
    check=True,
    # cwd="",
)
# %%
