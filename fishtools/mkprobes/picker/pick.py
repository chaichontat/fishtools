# %%
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import seaborn as sns

from oligocheck.external.external_data import ExternalData
from oligocheck.picker.codebook import CodebookPickerSingleCell
from oligocheck.picker.visualize import clustermap_

# %%
# counts = pl.read_parquet("data/fpkm/aging/aging_counts.parquet")
# obs = pl.read_parquet("data/fpkm/aging/obs.parquet")
# var = pl.read_parquet("data/fpkm/aging/var.parquet")

# %%
counts = pl.read_parquet("data/fpkm/embryo_raw.parquet")
# %%
# adata = sc.read_h5ad("data/fpkm/embryo_raw.h5ad")
# %%
# adata = sc.read_h5ad("data/fpkm/embryo_raw.h5ad")
# sc.pp.filter_cells(adata, min_genes=500)
# sc.pp.filter_genes(adata, min_cells=10)
# adata = adata[~adata.obs["New_cellType"].isin(["Doublet", "Low quality cells"])]
# sc.pp.normalize_total(adata, target_sum=1e4)
# %%
gtf = ExternalData(
    cache="data/mm39/gencode_vM32_transcripts.parquet",
    path="data/mm39/gencode.vM32.chr_patch_hapl_scaff.basic.annotation.gtf",
    fasta="data/mm39/combi.fa.gz",
)


def from_txt(path: str):
    return pl.Series(Path(path).read_text().splitlines())


# %%
combi = from_txt("constructed/combi.txt")
genes = pl.Series(Path("panels/motorcortex_converted.txt").read_text().splitlines())
gene_sm = pl.Series(Path("panels/motorcortex_smgenes.txt").read_text().splitlines())
# tricycle = pl.Series(Path("panels/tricycle_selected.txt").read_text().splitlines())
tricycle = pl.read_csv("data/tricycle.csv")
embryo = pl.read_parquet("panels/nmf.parquet")
eg = set(embryo.filter(pl.col("score") > 0.7)["gene"].unique())
# %%
ep = (
    set(from_txt("panels/publishedembryo.txt"))
    | set(from_txt("panels/birthdate_diff.txt"))
    | set(
        """Hes1
Sox2
Neurog2
Eomes
Btg2
Igfbpl1
Pcp4
Nrp1
Neurod2
Neurod1
Unc5d
Sema3c
Tbr1
Bcl11b
Tubb3
Fezf2
Sox5
Reln
Calb2
Emx2
Lhx5
Tle4
Ldb2
Tshz2
Etv1
Nr4a2
Rorb
Lpl
Ptn
Satb2
Cux1
Dlx1
Dlx2
Dlx5
Gad2
Lhx6
Cux2
Apoe
Aldoc
Slc1a3
Olig1
Pdgfra
Olig2
C1qb
C1qc
P2ry12
Aif1
Foxj1
Cldn5
Igfbp7
Rgs5
Col3a1
Lgals1
Lum
Kcnj8
Pdgfrb
Car2
Slc17a6
Neurod4
Dcc
Gsg1l
Rgs4
Lct
Smad3
Dsp
Grp
C1ql2
Icam5
Cemip
Lmo1
Calb1
Bhlhe22
Hemgn
Aqp4
""".splitlines()
    )
    | set(
        """Gad2,Emx1,Npy,Sst,Lhx6,Nxph1,Htr3a,Prox1,Cxcl14,Meis2,Etv1,Sp8,Sox2,Eomes,Nrp1,Tubb3,Satb2,Cux2,Bcl11b,Tle4,Reln,Dlx2,Pdgfra,Apoe,Cldn5,Aif1,Car2,Col1a1""".split(
            ","
        )
    )
    | set(tricycle[:125]["symbol"].to_list())
    | set("".split(" "))
)
c, _, _ = gtf.check_gene_names(ep)
# gdf = (
#     pl.DataFrame({"gene_id": list(var["var_names"])})
#     .join(gtf.gtf.unique("gene_id"), on="gene_id", how="left")
#     .fill_null("")
#     .select(pl.col("gene_name") + "_" + pl.col("gene_id"))["gene_name"]
# )


# %%
zach = set(Path("zach.txt").read_text().splitlines())
# ep |= zach
# adata.var_names_make_unique()
# counts = adata[:, list(set(c[0]) - {"Foxi1"})]
# %%
# selected.to_pandas().corrwith(
#     pd.get_dummies(adata.obs["New_cellType"].reset_index(drop=True)).astype(int), axis=0, numeric_only=True
# )


# pd.get_dummies(adata.obs["New_cellType"])

# %%
# genes = pl.Series(["Col2a1", "Fgfr3"])
selected = counts.select([pl.col(x) for x in ("^" + pl.Series(list(c)) + "$")])
# clustermap_(
#     pd.concat(
#         [selected.to_pandas(), pd.get_dummies(adata.obs["New_cellType"].reset_index(drop=True))], axis=1
#     )
# )

perc999 = (
    selected.select(pl.all().map(lambda x: np.percentile(x, 99.99)))
    .transpose(include_header=True, header_name="gene", column_names=["counts"])
    .sort("counts")
)

# perc999 = np.percentile(counts.X.A, axis=0, q=99.99)

# %%
perc999.to_pandas().hist(bins=50)
# %%
perc999.filter(pl.col("counts") > 24.5)

# %%
print(len(perc999.filter(pl.col("counts") < 24.5)))
perc999.filter(pl.col("counts") < 25).to_pandas().hist(bins=50)


# %%
def lt(n: float, prefix: str):
    perc999.filter(pl.col("counts") < n)[["gene"]].write_csv(f"{prefix}_{n}.txt", has_header=False)


lt(24.5, "cs")
# perc999.filter(pl.col("counts") < 60)[['gene']].write_csv('lt60.csv', has_header=False)
# %%

# adata.obs['New_cellType']

counts[:, perc999.filter(pl.col("counts") < 25)["gene"]].to_pandas().sum(axis=1).hist(bins=50)
# %%


dfg = pl.DataFrame(
    dict(
        age=adata.obs["orig_ident"].to_list(),
        ct=adata.obs["New_cellType"].to_list(),
        count=counts[:, perc999.filter(pl.col("counts") < 25)["gene"]].to_pandas().sum(axis=1),
    )
)


sns.set()
sns.FacetGrid(
    dfg.filter(pl.col("ct").str.contains("Intermediate progenitors")).to_pandas(), col="age", col_wrap=3
).map(sns.histplot, "count", bins=50)
# sns.histplot(
#     data=dfg.filter(pl.col("ct").str.contains("Apical progenitors")).to_pandas(),
#     x="count",
#     hue="age",
#     bins=50,
# )
# %%
perc999.filter(pl.col("counts") < 50)[["gene"]].sort("gene").write_csv("lt50.txt", has_header=False)
# %%

# %%
# %%
# tc = tricycle.join(perc999, left_on="symbol", right_on="gene")[["symbol", "max_pca_weight", "counts"]]

# %%

# %%
mhd = CodebookPickerSingleCell("static/mhd4_27bit.csv")


# %%
cl = counts[:, perc999.filter(pl.col("counts") < 50)["gene"]].to_numpy()
mhd.find_optimalish(cl, iterations=100)
# %%
mhd.gen_codebook(36).shape
# %%
mhd27 = np.loadtxt("static/mhd4_27bit.csv", delimiter=",", dtype=bool)
# %%
mhd27[mhd27[:, -3:].sum(axis=1) == 0].shape
# %%
