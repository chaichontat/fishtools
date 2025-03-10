# %%
import json
from pathlib import Path

import polars as pl

from fishtools.mkprobes.starmap.simple import generate_splint_padlock, process, run_pipeline
from fishtools.mkprobes.utils.sequtils import gen_idt

readouts = pl.read_csv("data/readout_ref_filtered.csv")

species = "mouse"
cwd = Path("merbenchmark")
codebook_path = Path(f"{cwd}/genes.json").resolve()
codebook: dict[str, list[int]] = json.loads(codebook_path.read_text())
# %%
run_pipeline(codebook_path, cwd, species)
# %%
df = process(codebook, cwd, n=16)
df = generate_splint_padlock(df, readouts)


# %%
def prepend_anchor(s: str):
    return "TGTTGATGAGGTGTTGATGAA" + s


out = df.with_columns(
    seq=pl.when(pl.col("name").str.starts_with("Spl")).then(pl.col("seq"))
    # .then(pl.col("seq").map_elements(prepend_anchor, return_dtype=pl.Utf8))
    .otherwise("/5Phos/" + pl.col("seq"))
)

# %%
print("\n".join(out["seq"]))
# %%
