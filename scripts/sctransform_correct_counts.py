from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse

DEFAULT_H5AD = Path("/fast2/cs_outputs/all_progenitors2.h5ad")
DEFAULT_INPUT_LAYER = "raw"
DEFAULT_BATCH_KEY = "dataset"
DEFAULT_OUTPUT_LAYER = "raw_sct_corrected"
DEFAULT_R_SCRIPT = Path(__file__).with_name("sctransform_correct_counts.R")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run R sctransform::vst + sctransform::correct_counts on an AnnData layer, "
            "then write corrected counts back to the same .h5ad."
        )
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD, help="Path to input/output .h5ad file.")
    parser.add_argument(
        "--input-layer",
        type=str,
        default=DEFAULT_INPUT_LAYER,
        help="Input counts layer in adata.layers.",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default=DEFAULT_BATCH_KEY,
        help="Batch key in adata.obs passed to sctransform as batch_var.",
    )
    parser.add_argument(
        "--output-layer",
        type=str,
        default=DEFAULT_OUTPUT_LAYER,
        help="Output layer name to store corrected counts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output layer.",
    )
    parser.add_argument(
        "--r-script",
        type=Path,
        default=DEFAULT_R_SCRIPT,
        help="Path to helper R script that runs sctransform.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Optional temporary directory for intermediate MatrixMarket files.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="Verbosity passed to sctransform::vst/correct_counts.",
    )
    return parser.parse_args()


def _to_csr_float32(matrix: object) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    dense = np.asarray(matrix, dtype=np.float32)
    return sparse.csr_matrix(dense)


def _validate_inputs(adata: ad.AnnData, *, input_layer: str, batch_key: str, output_layer: str, overwrite: bool) -> None:
    if input_layer not in adata.layers:
        raise KeyError(f"Missing input layer {input_layer!r}. Available layers: {list(adata.layers.keys())}")
    if batch_key not in adata.obs:
        raise KeyError(f"Missing batch key {batch_key!r} in adata.obs.")
    if adata.obs[batch_key].isna().any():
        raise ValueError(f"Batch key {batch_key!r} contains missing values in adata.obs.")
    if output_layer in adata.layers and not overwrite:
        raise ValueError(
            f"Output layer {output_layer!r} already exists. Use --overwrite or choose a different --output-layer."
        )


def _run_r_sctransform(
    *,
    r_script: Path,
    counts_path: Path,
    genes_path: Path,
    cells_path: Path,
    cell_attr_path: Path,
    output_path: Path,
    batch_key: str,
    verbosity: int,
) -> None:
    cmd = [
        "Rscript",
        str(r_script),
        "--counts",
        str(counts_path),
        "--genes",
        str(genes_path),
        "--cells",
        str(cells_path),
        "--cell-attr",
        str(cell_attr_path),
        "--batch-key",
        str(batch_key),
        "--out",
        str(output_path),
        "--verbosity",
        str(verbosity),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "R sctransform subprocess failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def main() -> None:
    args = _parse_args()

    if not args.h5ad.exists():
        raise FileNotFoundError(f"Input h5ad not found: {args.h5ad}")
    if not args.r_script.exists():
        raise FileNotFoundError(f"R helper script not found: {args.r_script}")

    adata = ad.read_h5ad(args.h5ad)
    _validate_inputs(
        adata,
        input_layer=args.input_layer,
        batch_key=args.batch_key,
        output_layer=args.output_layer,
        overwrite=args.overwrite,
    )

    counts_cxg = _to_csr_float32(adata.layers[args.input_layer])
    counts_gxc = counts_cxg.transpose().tocsc()
    genes = pd.Index(adata.var_names.astype(str), name="gene")
    original_cells = adata.obs_names.astype(str).to_numpy()
    batch = adata.obs[args.batch_key].astype(str).to_numpy()
    cells = pd.Index(
        [cell if cell.startswith(f"{dataset}_") else f"{dataset}_{cell}" for cell, dataset in zip(original_cells, batch)],
        name="cell",
    )

    if counts_cxg.shape != (len(cells), len(genes)):
        raise ValueError(
            f"Layer shape mismatch: layer={counts_cxg.shape}, obs={len(cells)}, var={len(genes)}."
        )

    cell_attr = pd.DataFrame({"cell": cells.to_numpy(), args.batch_key: batch})

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        counts_path = tmpdir / "counts_gxc.mtx"
        genes_path = tmpdir / "genes.tsv"
        cells_path = tmpdir / "cells.tsv"
        cell_attr_path = tmpdir / "cell_attr.tsv"
        corrected_path = tmpdir / "corrected_gxc.mtx"

        spio.mmwrite(str(counts_path), counts_gxc)
        genes.to_series(index=None).to_csv(genes_path, sep="\t", header=False, index=False)
        cells.to_series(index=None).to_csv(cells_path, sep="\t", header=False, index=False)
        cell_attr.to_csv(cell_attr_path, sep="\t", header=True, index=False)

        _run_r_sctransform(
            r_script=args.r_script,
            counts_path=counts_path,
            genes_path=genes_path,
            cells_path=cells_path,
            cell_attr_path=cell_attr_path,
            output_path=corrected_path,
            batch_key=args.batch_key,
            verbosity=args.verbosity,
        )

        corrected_gxc = spio.mmread(str(corrected_path))
        corrected_gxc = _to_csr_float32(corrected_gxc).transpose().tocsr()

    if corrected_gxc.shape != counts_cxg.shape:
        raise ValueError(
            f"Corrected layer shape mismatch: corrected={corrected_gxc.shape}, expected={counts_cxg.shape}."
        )

    adata.layers[args.output_layer] = corrected_gxc
    adata.write_h5ad(args.h5ad)

    print(
        f"Wrote corrected counts to {args.h5ad} in adata.layers[{args.output_layer!r}] "
        f"using batch key {args.batch_key!r}."
    )


if __name__ == "__main__":
    main()
