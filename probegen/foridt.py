# %%
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from fishtools import hp, rc
from fishtools.mkprobes.codebook.codebook import hash_codebook
from fishtools.mkprobes.starmap.starmap import test_splint_padlock

readouts = pl.read_csv("data/readout_ref_filtered.csv")

hhp = lambda x: hp(x, "dna")
species = "mouse"
cwd = "starhcrtest"
codebook = Path(f"{cwd}/genes.json").resolve()

# %%
genes = json.loads(codebook.read_text())
hsh = hash_codebook(genes)
with ThreadPoolExecutor() as executor:
    for gene in genes:
        executor.submit(
            subprocess.run, f"mkprobes candidates ../data/{species} -o output/ -g {gene}", shell=True, cwd=cwd
        )

# %%
with ThreadPoolExecutor() as executor:
    for gene in genes:
        executor.submit(subprocess.run(f"mkprobes screen output {gene}", shell=True, cwd=cwd))
# %%
with ThreadPoolExecutor() as executor:
    futs = []
    for gene in genes:
        futs.append(
            executor.submit(
                subprocess.run,
                f"mkprobes construct ../data/{species} output -g {gene} -c {codebook}",
                shell=True,
                cwd=cwd,
            )
        )
for fut in futs:
    fut.result()


# %%
def create_pad(seq: str, readout: str):
    arr = ["GTAGACTA", seq.lower(), readout, "ATTGGTTC"]
    out = "".join(arr)
    if len(out) > 60:
        raise ValueError("Too long")
    if len(out) > 57:
        return out

    arr.insert(2, "TACATA"[: 57 - len(out)])
    return "".join(arr)


def create_spl(seq: str):
    return seq.lower() + "TA GTCTAC GAACCA"


n = 16
dfs = {k: pl.read_parquet(f"{cwd}/output/{k}_final_{bits[0]}.parquet") for k, bits in genes.items()}
res = {}
for df in dfs:
    df_ = (
        dfs[df]
        .with_columns(
            hairpin=pl.max_horizontal(
                pl.col("cons_splint").map_elements(hhp, return_dtype=pl.Float32),
                pl.col("cons_pad").map_elements(hhp, return_dtype=pl.Float32),
            ),
            pos_start=pl.col("pos_start").list.get(0),
        )
        .filter(pl.col("pos_start") > 50)
        .filter(pl.col("splint").str.len_chars() > 17)
        .filter(pl.col("padlock").str.len_chars() < 24)
        .sort("hairpin", descending=False)[: n + 2]
        .with_columns(gene=pl.col("name").str.split("_").list.get(0))
        .sort("pos_start")
        .with_row_index("i", offset=0)
    )

    # Filter too close probes.
    for i in reversed(range(len(df_))):
        if df_[i, "pos_start"] - df_[i - 1, "pos_start"] < 50:
            df_ = df_.filter(pl.col("i") != i - 1)

    res[df] = df_.sample(min(n, len(df_)), seed=4)
    print(f"{df}: {len(res[df])}")


# print([res[r]["pos"] for r in res])
df = pl.concat([df.select(next(iter(dfs.values())).columns) for df in res.values()])
# res = [pick(i) for i in range(200)]
# lowest = min(res)
# idx = res.index(lowest)


# %%
out = []
for i, row in enumerate(df.iter_rows(named=True)):
    spl = create_spl(rc(row["splint"])).replace(" ", "")
    pad = create_pad(rc(row["padlock"]), readouts[row["code"] - 1, "seq"]).replace(" ", "")
    assert len(spl) < 60, "spl"
    assert len(pad) < 60, f"pad {len(row['padlock'])}"
    assert test_splint_padlock(spl, pad)
    out.append({"name": f"Spl-{row['gene']}-{i}-R{row['code']}", "seq": spl})
    out.append({"name": f"Pad-{row['gene']}-{i}-R{row['code']}", "seq": pad})

out = pl.DataFrame(out)
# %%
# %%
for row in out.iter_rows():
    print(f"{row[1]}")


# %%
# %%


# hhp = lambda x: hp(x, "dna")
# # %%
# genes = json.loads(Path("alina/braingenes.json").read_text())
# # [
# #     subprocess.run(
# #         f"mkprobes construct ../data/mouse output/ -g {gene} -c genes.json", shell=True, cwd="alina"
# #     )
# #     for gene in genes
# # ]
# # %%
# dfs = {k: pl.read_parquet(f"alina/output/{k}_final_792398.parquet") for k in genes}
# res = {}
# for df in dfs:
#     res[df] = (
#         dfs[df]
#         .with_columns(
#             hairpin=pl.max_horizontal(pl.col("cons_splint").apply(hhp), pl.col("cons_pad").apply(hhp))
#         )
#         .filter(pl.col("pos") > 50)
#         .sort("hairpin")[:5]
#         .with_row_count("i", offset=1)
#         .with_columns(gene=pl.col("name").str.split("_").list.get(0))
#     )
# df = pl.concat(res.values())

# [res[r]["pos"] for r in res]
# # res = [pick(i) for i in range(200)]
# # lowest = min(res)
# # idx = res.index(lowest)

# # %%
# for row in df.iter_rows(named=True):
#     print(f"STARSp-{row['gene']}-{row['i']}\t{row['cons_splint']}\t\t25nm\tSTD")

# for row in df.iter_rows(named=True):
#     print(f"STARPad-{row['gene']}-{row['i']}\t{row['cons_pad']}\t\t25nm\tSTD")

# import pandas as pd

# %%
# %%
from fishtools.mkprobes.utils.sequtils import gen_plate

# %%
gen_plate("STARSp-Alina-Brain", [row["cons_pad"] for row in df.iter_rows(named=True)])

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter("alina/braingenes.xlsx", engine="xlsxwriter")

# Write each dataframe to a different worksheet.
gen_plate(
    [f"STARPad-{row['gene']}-{row['i']}" for row in df.iter_rows(named=True)],
    [row["cons_pad"] for row in df.iter_rows(named=True)],
).to_pandas().to_excel(writer, sheet_name="STARPad-Alina-Brain", index=False)

gen_plate(
    [f"STARSp-{row['gene']}-{row['i']}" for row in df.iter_rows(named=True)],
    [row["cons_splint"] for row in df.iter_rows(named=True)],
).to_pandas().to_excel(writer, sheet_name="STARSp-Alina-Brain", index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
# %%
