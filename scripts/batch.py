# %%
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import polars as pl

from fishtools.mkprobes import get_candidates, run_screen
from fishtools.mkprobes.ext.external_data import Dataset


def run_gene(gene: str, transcript: str):
    print(f"running {gene}")
    get_candidates(Dataset("data/human"), transcript=transcript, output="output/", ignore_revcomp=True)
    run_screen("output/", gene, minimum=72, restriction=["KpnI"])


# @flow
def run():
    genes = pl.read_csv("bgcb_fixed.csv")[1:][["name", "id"]].filter(
        ~pl.col("name").str.starts_with("Blank") & pl.col("id").str.starts_with("ENST")
    )
    context = get_context("forkserver")
    with ProcessPoolExecutor(16, mp_context=context) as exc:
        futs = {
            (name, id): exc.submit(run_gene, gene=name, transcript=id)
            for name, id in genes[["name", "id"]].iter_rows()
            if not len(list(Path("output").glob(f"{name}*_screened.parquet")))
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
