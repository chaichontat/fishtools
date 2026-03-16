from __future__ import annotations

from pathlib import Path


def safe_gene_name(gene: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in gene)


def build_fit_map(fits_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for fit_path in sorted(fits_dir.glob("*.gam.rds")):
        base = fit_path.name
        if "_" in base:
            safe = base.split("_", 1)[1].removesuffix(".gam.rds")
        else:
            safe = base.removesuffix(".gam.rds")
        if safe not in out:
            out[safe] = fit_path
    if not out:
        raise FileNotFoundError(f"No .gam.rds files found under {fits_dir}")
    return out
