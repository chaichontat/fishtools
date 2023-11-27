# %%
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from fishtools import rc
from fishtools.mkprobes.candidates import get_candidates
from fishtools.mkprobes.codebook.codebook import CodebookPicker
from fishtools.mkprobes.codebook.finalconstruct import assign_overlap, construct
from fishtools.mkprobes.ext.external_data import Dataset
from fishtools.mkprobes.ext.fix_gene_name import check_gene_names
from fishtools.mkprobes.genes.chkgenes import get_transcripts
from fishtools.mkprobes.screen import run_screen
from fishtools.mkprobes.starmap.starmap import split_probe, test_splint_padlock

# %%


# # # %%
#
#


# %%
def parse(df: pl.DataFrame) -> pl.DataFrame:
    if len(res := df.filter(pl.col("tag") == "Ensembl_canonical")):
        return res
    return df.sort("annotation", descending=False)[0]


# %%

mode = "genestar"


def run_gene(mode: str, gene: str, **kwargs):
    ds = Dataset("../data/mouse")
    restriction = ["BamHI", "KpnI"]

    try:
        # get_candidates(ds, gene=gene, output="output/", ignore_revcomp=True, **kwargs)
        # time.sleep(1)
        # run_screen("output/", gene, minimum=20, restriction=restriction, overwrite=True, maxoverlap=0)
        construct(
            ds,
            "output/",
            transcript=gene,
            codebook=json.loads(Path(f"{mode}.json").read_text()),
            restriction=restriction,
            target_probes=20,
        )
    except Exception as e:
        # df = pl.read_parquet("output/" + gene + "_screened_ol-2_BamHIKpnI.parquet")
        # print(df)
        raise Exception(gene) from e

    # # print(f"running {gene}")


def run():
    context = get_context("forkserver")
    with ProcessPoolExecutor(16, mp_context=context) as exc:
        futs = {
            (name): exc.submit(run_gene, gene=name, mode=mode)
            for name in Path(f"{mode}.tss.txt").read_text().splitlines()
            # if not Path(f"output/{name}_screened_ol-2_KpnI.parquet").exists()
            # or time.time() - os.path.getmtime(Path(f"output/{name}_screened_ol-2_KpnI.parquet")) > 3600
            # if pl.read_parquet(f"output/{name}_final_KpnI.parquet")["seq"].str.contains("N").any()
        }
        # futs = {
        #     (name, id): exc.submit(
        #         run_gene, gene=gene_name_mapping.get(name, name), transcript=id, **special.get(name, {})
        #     )
        #     for name, id in max_tss[["gene_name", "transcript_id"]].iter_rows()
        #     if not len(list(Path("output").glob(f"{name}*_final.parquet")))
        # }
        for fut in as_completed(futs.values()):
            try:
                fut.result()
            except Exception as e:
                for x, f in futs.items():
                    if f == fut:
                        print(x)
                        # break
                raise e


# %%

# dfs = pl.concat([pl.read_parquet(f"output/{m}_final.parquet") for m in max_tss["gene_name"]])
# counts = dfs.groupby("gene").agg(pl.count())

# counts.filter(pl.col("count") < 40)

# %%
if __name__ == "__main__":
    run()
# # %%
# pl.Config.set_fmt_str_lengths(65)
# pl.Config.set_tbl_rows(100)

# pl.read_parquet(next(Path("output").glob("Zfp386*_crawled.parquet")))

# pl.read_csv(next(Path("output").glob("Zfp386*_acceptabletss.csv")))

# %%

# genes = Path("synapse500.converted.txt").read_text().splitlines()

# ols = {}
# for gene in genes:
#     try:
#         ols[gene] = assign_overlap("output/", gene)
#     except FileNotFoundError:
#         ...

# dfs = []
# for gene, ol in ols.items():
#     df = pl.read_parquet(
#         f"output/{gene}_screened_ol{ol}.parquet" if ol else f"output/{gene}_screened.parquet"
#     )
#     dfs.append(df)

# dfs = pl.concat(dfs)
# # %%
# Path("synapse_filtered.txt").write_text(
#     "\n".join(
#         filtered := dfs.groupby("gene").agg(pl.count()).filter(pl.col("count") > 48)["gene"].sort().to_list()
#     )
# )


# %%


# %%

# if __name__ != "__main__":
