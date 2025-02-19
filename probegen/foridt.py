# %%
import json
from pathlib import Path

import polars as pl

from fishtools.mkprobes.starmap.simple import process, run_pipeline

readouts = pl.read_csv("data/readout_ref_filtered.csv")

species = "mouse"
cwd = Path("starhcrtest")
codebook_path = Path(f"{cwd}/genes.json").resolve()
codebook: dict[str, list[int]] = json.loads(codebook_path.read_text())

run_pipeline(codebook_path, cwd, species)
df = process(codebook, readouts, cwd, n=16)

# %%
