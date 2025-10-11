from pathlib import Path

import polars as pl
from click.testing import CliRunner

from fishtools.preprocess.spots.align_prod import spots as spots_cli


def _make_workspace(tmp_path: Path, rois: list[str]) -> None:
    # Create minimal directories so Workspace discovers ROIs
    for roi in rois:
        (tmp_path / f"R1--{roi}").mkdir(parents=True, exist_ok=True)
    # Ensure output location exists
    (tmp_path / "analysis" / "output").mkdir(parents=True, exist_ok=True)


def _write_parquet(tmp_path: Path, roi: str, codebook_sanitized: str) -> Path:
    spots = pl.DataFrame({
        "x": [10.0, 20.0, 30.0],
        "y": [5.0, 15.0, 25.0],
        "target": ["GeneA", "GeneB", "Blank-1"],
    })
    out = tmp_path / "analysis" / "output" / f"{roi}+{codebook_sanitized}.parquet"
    spots.write_parquet(out)
    return out


def test_spots_plotall_single_roi(tmp_path: Path) -> None:
    _make_workspace(tmp_path, ["roiA"])  # ROI discovery
    _write_parquet(tmp_path, "roiA", "cs_base")

    runner = CliRunner()
    res = runner.invoke(
        spots_cli,
        [
            "plotall",
            str(tmp_path),
            "roiA",
            "--codebook",
            "cs-base",  # will be sanitized to cs_base
            "--threads",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    out_png = tmp_path / "analysis" / "output" / "plots" / "plotall--roiA+cs_base.png"
    assert out_png.exists()


def test_spots_plotall_all_rois(tmp_path: Path) -> None:
    _make_workspace(tmp_path, ["roiA", "roiB"])  # ROI discovery
    _write_parquet(tmp_path, "roiA", "cs_base")
    _write_parquet(tmp_path, "roiB", "cs_base")

    runner = CliRunner()
    res = runner.invoke(
        spots_cli,
        [
            "plotall",
            str(tmp_path),  # omit ROI to trigger default "*"
            "--codebook",
            "cs-base",
            "--threads",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    outA = tmp_path / "analysis" / "output" / "plots" / "plotall--roiA+cs_base.png"
    outB = tmp_path / "analysis" / "output" / "plots" / "plotall--roiB+cs_base.png"
    assert outA.exists() and outB.exists()


def test_spots_plotall_default_roi_all(tmp_path: Path) -> None:
    # Same as all_rois, but explicitly verify that omitting the ROI works
    _make_workspace(tmp_path, ["roiX"])  # ROI discovery
    _write_parquet(tmp_path, "roiX", "cs_base")

    runner = CliRunner()
    res = runner.invoke(
        spots_cli,
        [
            "plotall",
            str(tmp_path),
            "--codebook",
            "cs-base",
            "--threads",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    outX = tmp_path / "analysis" / "output" / "plots" / "plotall--roiX+cs_base.png"
    assert outX.exists()


def test_spots_plotall_max_per_plot(tmp_path: Path) -> None:
    # Create a workspace with a single ROI and many genes to trigger splitting
    _make_workspace(tmp_path, ["roiA"])  # ROI discovery
    # Create 25 genes so with max-per-plot=20 we get 2 parts
    n_genes = 25
    genes = [f"Gene{i:03d}" for i in range(n_genes)]
    # 3 points per gene
    x = [float(i % 50) for i in range(n_genes * 3)]
    y = [float((i * 2) % 50) for i in range(n_genes * 3)]
    targets = list(genes for _ in range(3))  # 3 lists of genes
    import itertools

    df = pl.DataFrame({
        "x": x,
        "y": y,
        "target": list(itertools.chain.from_iterable(targets)),
    })
    out_parq = tmp_path / "analysis" / "output" / "roiA+cs_base.parquet"
    df.write_parquet(out_parq)

    runner = CliRunner()
    res = runner.invoke(
        spots_cli,
        [
            "plotall",
            str(tmp_path),
            "roiA",
            "--codebook",
            "cs-base",
            "--threads",
            "1",
            "--max-per-plot",
            "20",
        ],
    )
    assert res.exit_code == 0, res.output
    out1 = tmp_path / "analysis" / "output" / "plots" / "plotall--roiA+cs_base.1.png"
    out2 = tmp_path / "analysis" / "output" / "plots" / "plotall--roiA+cs_base.2.png"
    assert out1.exists() and out2.exists()
