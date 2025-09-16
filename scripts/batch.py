# %%
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from fishtools.mkprobes import get_candidates, run_screen
from fishtools.mkprobes.codebook.finalconstruct import construct
from fishtools.mkprobes.ext.dataset import ReferenceDataset as Dataset
from fishtools.mkprobes.utils.sequtils import reverse_complement


def run_gene(transcript: str):
    print(f"running {transcript}")
    restriction = ["KpnI"]
    ds = Dataset("data/human")
    get_candidates(
        ds := Dataset("data/human"), gene=transcript, output="output/", ignore_revcomp=True, realign=True
    )
    run_screen("output/", transcript, minimum=72, restriction=restriction, overwrite=True)
    construct(
        ds,
        "output/",
        transcript=transcript,
        codebook=json.loads(Path("bg_cb.json").read_text()),
        restriction=restriction,
    )


# %%
# genes = pl.read_csv("bgcb_fixed.csv")[1:].filter(
#     ~pl.col("name").str.starts_with("Blank") & pl.col("id").str.starts_with("ENST")
# )
# ds = Dataset("data/human")
# res = ds.ensembl.batch_convert(genes["id"].to_list(), src="transcript_id", dst="transcript_name")
# Path("bg_tsname.txt").write_text("\n".join(res))
# # %%
# codebook = {
#     ts: (np.flatnonzero(cb) + 1).tolist()
#     for ts, cb in zip(res, genes[:, 3:].with_columns(pl.col("Aux4").cast(pl.Int64)).fill_null(0).to_numpy())
# }

# Path("bg_cb.json").write_text(json.dumps(codebook, default=int))
tss = Path("bg_tsname.txt").read_text().splitlines()


# %%


# @flow
def run():
    context = get_context("forkserver")
    with ProcessPoolExecutor(24, mp_context=context) as exc:
        futs = {
            name: exc.submit(run_gene, transcript=name)
            for name in tss
            if pl.read_parquet(f"output/{name}_final_KpnI.parquet")["seq"].str.contains("N").any()
        }
        for fut in as_completed(futs.values()):
            try:
                fut.result()
            except Exception as e:
                for x, f in futs.items():
                    if f == fut:
                        print(x)
                        break
                raise e


if __name__ == "__main__":
    run()

# %
# %%

c = json.loads(Path("bg_cb.json").read_text())
dfs = {}
for m in tss:
    dfs[m] = (
        pl.read_parquet(f"output/{m}_final_KpnI.parquet")
        .sample(shuffle=True, seed=4, fraction=1)
        .sort("priority")[:71]
    )
    assert set(c[m]) == set(dfs[m]["code1"]) | set(dfs[m]["code2"]), (
        set(c[m]),
        set(dfs[m]["code1"]) | set(dfs[m]["code2"]),
    )

dfs = pl.concat(dfs.values())
counts = dfs.group_by("gene").agg(pl.count())

ok = counts.filter(pl.col("count").gt(39) | pl.col("gene").eq("HES3"))["gene"]
dfs = dfs.filter(pl.col("gene").is_in(ok))
# # %%

# %%
# %%

# %%
headers = pl.read_csv("/home/chaichontat/fishtools/fishtools/mkprobes/codebook/headers.csv", separator="\t")


header = reverse_complement(headers[0, "header"]) + "GGTACC"
footer = "TATTTCCC" + headers[0, "footer"]


# %%
def backfill(seq: str, target: int = 148):
    return "TTCCACTAACTCACTTCTTACC"[: max(0, target - len(seq))] + seq


# %%
t7 = "TAATACGACTCACTATAGGG"
cons = dfs.with_columns(constructed=header + pl.col("seq") + footer).with_columns(
    constructed=pl.col("constructed").apply(backfill)
)
# %%
assert cons["constructed"].str.contains("GGTACC").all()
assert cons["constructed"].str.contains(reverse_complement(t7)).all()
assert not cons["constructed"].str.contains("N").any()
# %%
Path("bg_out.csv").write_text("\n".join(cons["constructed"]))
# %%
