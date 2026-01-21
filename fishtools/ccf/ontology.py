from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd


CCFTermKind = Literal["auto", "id", "acronym", "name"]


def _categorical_mask(values: object, *, predicate: Callable[[str], bool]) -> np.ndarray:
    cat = pd.Categorical(values)
    codes = cat.codes
    cats = [str(v) for v in cat.categories]
    keep = np.asarray([bool(predicate(v)) for v in cats], dtype=bool)
    out = np.zeros(codes.shape, dtype=bool)
    valid = codes >= 0
    out[valid] = keep[codes[valid]]
    return out


def mask_ccf_subtree(
    adata: ad.AnnData,
    term: int | str,
    *,
    kind: CCFTermKind = "auto",
    obsm_key: str = "ccf",
) -> np.ndarray:
    """Return a boolean mask selecting cells under an ontology subtree.

    This uses the per-cell CCF ontology paths stored in ``adata.obsm[obsm_key]``.
    A cell is selected if `term` matches any node in its ontology path, which
    includes the node itself and all descendants.

    Expected schema: ``adata.obsm[obsm_key]`` is a DataFrame created by
    `ccf/atlas_annotate_h5ad.py` with at least:
    - ``path_ids`` for kind="id"
    - ``path_acronyms`` for kind="acronym"
    - ``path_names`` for kind="name"
    """

    if obsm_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{obsm_key!r}].")
    table = adata.obsm[obsm_key]
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Expected adata.obsm[{obsm_key!r}] to be a DataFrame, got {type(table).__name__}.")

    kind_norm = str(kind).lower()
    if kind_norm not in {"auto", "id", "acronym", "name"}:
        raise ValueError(f"Invalid kind={kind!r}. Expected 'auto'|'id'|'acronym'|'name'.")

    if kind_norm == "auto":
        if isinstance(term, int):
            kind_norm = "id"
        else:
            s = str(term).strip()
            if s.isdigit():
                kind_norm = "id"
            else:
                # Users often provide either acronyms ("Pal") or human names ("pallium");
                # treat auto as a union over both string namespaces.
                mask_acr = mask_ccf_subtree(adata, s, kind="acronym", obsm_key=obsm_key)
                mask_name = mask_ccf_subtree(adata, s, kind="name", obsm_key=obsm_key)
                return np.asarray(mask_acr, dtype=bool) | np.asarray(mask_name, dtype=bool)

    if kind_norm == "id":
        term_id = int(term)
        if term_id == 0:
            if "id" not in table.columns:
                raise KeyError(f"Missing 'id' column in adata.obsm[{obsm_key!r}].")
            return table["id"].to_numpy(dtype=np.int64, copy=False) == 0
        if "path_ids" not in table.columns:
            raise KeyError(f"Missing 'path_ids' column in adata.obsm[{obsm_key!r}].")

        def pred(s: str) -> bool:
            if not s:
                return False
            try:
                ids = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON path_ids entry: {s!r}") from exc
            if not isinstance(ids, list):
                raise ValueError(f"Invalid path_ids entry: expected JSON list, got {type(ids).__name__}.")
            return term_id in {int(v) for v in ids}

        return _categorical_mask(table["path_ids"], predicate=pred)

    if kind_norm == "acronym":
        q = str(term).strip().lower()
        if q == "background":
            if "id" not in table.columns:
                raise KeyError(f"Missing 'id' column in adata.obsm[{obsm_key!r}].")
            return table["id"].to_numpy(dtype=np.int64, copy=False) == 0
        if "path_acronyms" not in table.columns:
            raise KeyError(f"Missing 'path_acronyms' column in adata.obsm[{obsm_key!r}].")

        def pred(s: str) -> bool:
            if not s:
                return False
            return q in (seg.lower() for seg in s.split("/"))

        return _categorical_mask(table["path_acronyms"], predicate=pred)

    q = str(term).strip().lower()
    if q == "background":
        if "id" not in table.columns:
            raise KeyError(f"Missing 'id' column in adata.obsm[{obsm_key!r}].")
        return table["id"].to_numpy(dtype=np.int64, copy=False) == 0
    if "path_names" not in table.columns:
        raise KeyError(f"Missing 'path_names' column in adata.obsm[{obsm_key!r}].")

    def pred(s: str) -> bool:
        if not s:
            return False
        return q in (seg.lower() for seg in s.split("/"))

    return _categorical_mask(table["path_names"], predicate=pred)


def filter_ccf_subtree(
    adata: ad.AnnData,
    term: int | str,
    *,
    kind: CCFTermKind = "auto",
    obsm_key: str = "ccf",
) -> ad.AnnData:
    """Subset AnnData to cells in a CCF ontology subtree (includes descendants)."""

    mask = mask_ccf_subtree(adata, term, kind=kind, obsm_key=obsm_key)
    return adata[mask]
