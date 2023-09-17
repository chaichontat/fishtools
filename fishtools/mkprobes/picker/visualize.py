# %%
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
import seaborn as sns
from oligocheck.boilerplate import copy_signature


def clustermap_(df: pd.DataFrame, **kwargs: Any):
    default_kw = dict(figsize=(20, 20), cmap="turbo")
    corr = df.corr() ** 2
    y = np.nan_to_num(corr, nan=0)
    np.fill_diagonal(y, 1)

    linkage = hc.linkage(
        np.nan_to_num(sp.distance.squareform(1 - corr, checks=False), nan=1),
        method="single",
    )
    res = sns.clustermap(corr, row_linkage=linkage, col_linkage=linkage, **(default_kw | kwargs))
    return corr, res, cast(npt.NDArray[np.int32], res.dendrogram_row.reordered_ind)


@copy_signature(sns.heatmap)
def heatmap_(corr: npt.NDArray[Any], **kwargs: Any):
    default_kw = dict(xticklabels=True, yticklabels=True, square=True, cmap="turbo")
    return sns.heatmap(corr, **{**default_kw, **kwargs})
