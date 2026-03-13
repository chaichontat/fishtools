#%%
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import sklearn.neighbors
import STAGATE_pyG
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

input_path = Path("/fast2/cs_outputs/all.h5ad")
output_path = input_path.with_name(f"{input_path.stem}.stagate.h5ad")
# %%


def Cal_Spatial_Net(adata, rad_cutoff=None, obsm="spatial", k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm[obsm])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    avg_neighbors = Spatial_Net.shape[0] / adata.n_obs
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        if model == 'Radius':
            print(f'Radius cutoff {rad_cutoff} -> {avg_neighbors:.4f} neighbors per cell on average.')
        else:
            print('%.4f neighbors per cell on average.' % (avg_neighbors,))

    adata.uns['Spatial_Net'] = Spatial_Net


def train_stagate_batched(
    adata,
    *,
    obsm="spatial_trans",
    batch_key=("dataset", "roi", "ccf_adjusted"),
    batch_size=1,
    n_epochs=1000,
    lr=0.0005,
    weight_decay=1e-4,
    hidden_dims=(512, 30),
    gradient_clipping=5.0,
    random_seed=0,
    rad_cutoff=50,
):
    """Train STAGATE on slide-local graphs with a shared model and shared latent space."""

    import random

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if isinstance(batch_key, str):
        batch_cols = [batch_key]
    else:
        batch_cols = list(batch_key)
    missing_batch = adata.obs[batch_cols].isna()
    if missing_batch.to_numpy().any():
        missing_counts = missing_batch.sum(axis=0)
        missing_summary = ", ".join(
            f"{col}={int(count)}" for col, count in missing_counts.items() if count > 0
        )
        raise ValueError(
            "Cells with missing batch_key values cannot be batched for shared STAGATE training. "
            f"Fill or filter them first. Missing counts: {missing_summary}."
        )

    batch_list = []
    for _, batch_obs in adata.obs.groupby(batch_cols, observed=True):
        temp_adata = adata[batch_obs.index].copy()
        if temp_adata.n_obs > 0:
            batch_list.append(temp_adata)
    if not batch_list:
        raise ValueError(f"No non-empty slide batches found for {batch_cols}.")
    batch_sizes = np.array([temp_adata.n_obs for temp_adata in batch_list], dtype=np.int64)
    logger.info(
        f"Batching by {batch_cols}: {len(batch_list)} groups, "
        f"min={batch_sizes.min()}, median={int(np.median(batch_sizes))}, max={batch_sizes.max()} cells."
    )

    data_list = []
    total_edges = 0
    total_cells = 0
    for i, temp_adata in enumerate(batch_list, start=1):
        Cal_Spatial_Net(temp_adata, model="Radius", rad_cutoff=rad_cutoff, obsm=obsm, verbose=False)
        total_edges += temp_adata.uns['Spatial_Net'].shape[0]
        total_cells += temp_adata.n_obs
        data_list.append(STAGATE_pyG.Transfer_pytorch_Data(temp_adata))
        if i == 1 or i == len(batch_list) or i % 10 == 0:
            logger.info(f"Prepared graph {i}/{len(batch_list)} with {temp_adata.n_obs} cells.")
    if total_cells > 0:
        logger.info(
            "Radius cutoff %.4f produced %.4f average neighbors per cell across all cells.",
            rad_cutoff,
            total_edges / total_cells,
        )

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    model = STAGATE_pyG.STAGATE(hidden_dims=[data_list[0].x.shape[1], *hidden_dims]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info(f"Starting training for {n_epochs} epochs with radius={rad_cutoff} and batch_size={batch_size}.")
    last_loss = None
    for epoch in tqdm(range(1, n_epochs + 1)):
        for batch in loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            z, out = model(batch.x, batch.edge_index)
            loss = F.mse_loss(batch.x, out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        if epoch == 1 or epoch == n_epochs or epoch % 50 == 0:
            logger.info(f"Epoch {epoch}/{n_epochs} loss={last_loss:.6f}")

    model.eval()
    embedding = np.zeros((adata.n_obs, hidden_dims[-1]), dtype=np.float32)
    with torch.inference_mode():
        for i, temp_adata in enumerate(batch_list, start=1):
            temp_data = STAGATE_pyG.Transfer_pytorch_Data(temp_adata).to(device)
            z, _ = model(temp_data.x, temp_data.edge_index)
            embedding[adata.obs_names.get_indexer(temp_adata.obs_names)] = z.to('cpu').detach().numpy()
            if i == 1 or i == len(batch_list) or i % 10 == 0:
                logger.info(f"Embedded batch {i}/{len(batch_list)}.")

    adata.obsm['STAGATE'] = embedding
    logger.info(f"Finished STAGATE embedding with shape {adata.obsm['STAGATE'].shape}.")
    return adata

def main() -> None:
    adata = ad.read_h5ad(input_path)
    mask = (
        np.isfinite(adata.obs['ap'])
        & np.isfinite(adata.obs['ml'])
        & np.isfinite(adata.obs['r_um'])
    )
    adata = adata[mask].copy()
    adata.obs_names_make_unique()
    logger.info(f"Training input after filtering: {adata.n_obs} cells x {adata.n_vars} genes.")
    #%%
    adata = train_stagate_batched(adata, obsm='spatial_trans', rad_cutoff=40)
    adata.write_h5ad(output_path)
    logger.info(f"Wrote STAGATE results to {output_path}")


if __name__ == "__main__":
    main()
# %%
# rsc.pp.neighbors(adata, use_rep='STAGATE')
# #%%
# rsc.tl.umap(adata, n_components=2)
# # %%
# sc.tl.leiden(adata, resolution=0.5, n_iterations=2, flavor="igraph")
# # %%
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
# umap3d = adata.obsm['X_umap']
# for color_key in ["ap", "ml", "r_um"]:
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     vals = adata.obs[color_key].values
#     sc = ax.scatter(umap3d[:, 0], umap3d[:, 1], umap3d[:, 2], c=vals, cmap="turbo", s=1, alpha=0.7)
#     plt.colorbar(sc, ax=ax, label=color_key)
#     ax.set_title(color_key)
#     plt.tight_layout()
#     plt.show()

# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
# leiden_cats = adata.obs['leiden'].astype('category')
# codes = leiden_cats.cat.codes.values
# sc_plot = ax.scatter(umap3d[:, 0], umap3d[:, 1], umap3d[:, 2], c=codes, cmap="tab20", s=1, alpha=0.7)
# plt.colorbar(sc_plot, ax=ax, label='leiden')
# ax.set_title('leiden')
# plt.tight_layout()
# plt.show()
# # %%
# sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
# # %%
# plot_ranked_genes(adata)
# # %%
# # fig,ax=plt.subplots(figsize=(18,18), dpi=300)
# sc.pl.embedding(adata, basis='umap', color=['leiden', 'Eomes', 'Pax6', 'tricycle'], cmap="CMRmap_r")
# # %%
# adata.obsm['AP_ML_um_jitter'] = adata.obsm['AP_ML_um'] + np.random.default_rng(0).normal(0, 50, adata.obsm['AP_ML_um'].shape)
# sc.pl.embedding(adata, basis='AP_ML_um_jitter', color='leiden')
# %%
