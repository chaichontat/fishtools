# %%
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

overwrite = True

if overwrite or not Path("starwork2/alz/combi.h5ad").exists():
    path = Path("starwork2/alz")

    names = set("_".join(name.stem.split("_")[:-1]) for name in path.glob("*.mtx.gz"))

    def move(name: str):
        files = ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]
        (path / name).mkdir(exist_ok=True)
        for file in files:
            shutil.move(path / f"{name}_{file}", path / name / file)

    [move(name) for name in names]

    names = set(p.name for p in path.iterdir() if p.is_dir())
    adatas: list[ad.AnnData] = []
    for name in names:
        print(name)
        adatas.append(sc.read_10x_mtx(path / name))
        adata = adatas[-1]
        adata.obs["sample"] = name
        adata.X = sparse.csr_matrix(adata.X)

    adata = ad.concat(adatas)
    adata.obs_names_make_unique()
    adata.var = adatas[0].var
    adata.write_h5ad("starwork2/alz/combi.h5ad")

# %%
adata = sc.read_h5ad("starwork2/alz/combi.h5ad")


# %%
genes = json.loads(Path("starwork2/10xhuman.json").read_text())
genes = ["-".join(gene.split("-")[:-1]) for gene in genes]


# %%
def get_percentile(gene: str, p: float = 99.9) -> int:
    try:
        return np.percentile(adata[:, gene].X.A, p)
    except KeyError:
        return 0


# %%
gs = json.loads(Path("starwork2/laihuman.json").read_text())
gs = ["-".join(g.split("-")[:-1]) for g in gs]

vs = {g: get_percentile(g) for g in gs}
vs = {k: vs[k] for k in sorted(vs, key=vs.get)}


# %%
