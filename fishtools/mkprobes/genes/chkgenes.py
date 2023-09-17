import json
import re
from pathlib import Path
from typing import Literal

import polars as pl
import rich_click as click
from loguru import logger as log

from fishtools.utils.pretty_print import jprint

from ..ext.external_data import Dataset


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("genes", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
def chkgenes(path: Path, genes: Path):
    """Validate/check that gene names are canonical in Ensembl"""
    from ..ext.fix_gene_name import check_gene_names

    ds = Dataset(path)
    gs: list[str] = re.split(r"[\s,]+", genes.read_text())
    if not gs:
        raise ValueError("No genes provided")

    gs = list(filter(lambda x: x, gs))
    if len(gs) != len(s := set(gs)):
        [gs.remove(x) for x in s]
        log.warning(f"Non-unique genes found: {', '.join(gs)}.\n")
        path.with_suffix(".unique.txt").write_text("\n".join(sorted(list(s))))
        log.error(f"Unique genes written to {path.with_suffix('.unique.txt')}.\n")
        log.critical("Aborting.")
        return

    converted, mapping, no_fixed_needed = check_gene_names(ds.ensembl, gs)
    if mapping:
        log.info("Mappings:")
        jprint(mapping)
        path.with_suffix(".mapping.json").write_text(json.dumps(mapping))
        path.with_suffix(".converted.txt").write_text("\n".join(sorted(converted)))
    elif not no_fixed_needed:
        log.warning("Some genes cannot be found.")
    else:
        log.info(f"{len(s)} genes checked out. No changes needed")


def _gettranscript(
    path: Path,
    gene: str,
    mode: Literal["gencode", "ensembl", "canonical", "appris", "apprisalt"] = "canonical",
) -> pl.DataFrame:
    """Get transcript ID from gene name or gene ID"""
    dataset = Dataset(Path(path))
    gene_id = gene if gene.startswith("ENSG") else dataset.ensembl.gene_to_eid(gene)
    del gene

    log.info(f"Gene ID: {gene_id}")

    ensembl = dataset.ensembl.filter(pl.col("gene_id") == gene_id)[
        ["gene_name", "gene_id", "transcript_name", "transcript_id"]
    ]
    ensembl = ensembl.join(
        dataset.appris.filter(pl.col("gene_id") == gene_id)[["transcript_id", "annotation"]],
        on="transcript_id",
        how="left",
    ).sort("transcript_name")
    appris = ensembl.filter(pl.col("annotation").is_not_null())

    # match mode:
    #     case "canonical":
    #         return [get_ensembl("output/", gene_id)["canonical_transcript"].split(".")[0]]
    #     case "gencode":
    #         return dataset.gencode.filter(pl.col("gene_id") == gene_id)["transcript_id"].to_list()
    #     case "ensembl":
    #         return dataset.ensembl.filter(pl.col("gene_id") == gene_id)["transcript_id"].to_list()
    #     case "appris":
    #         return appris.filter(pl.col("annotation").str.contains("PRINCIPAL"))["transcript_id"].to_list()
    #     case "apprisalt":
    #         return appris["transcript_id"].to_list()
    log.info(ensembl)
    log.info(appris)
    if len(principal := appris.filter(pl.col("annotation").str.contains("PRINCIPAL"))):
        log.info("Principal transcripts: " + "\n".join(principal["transcript_id"]))

    # tss_gencode = set(dataset.gencode.filter(pl.col("gene_id") == gene_id)["transcript_id"])
    # tss_allofgene = set(dataset.ensembl.filter(pl.col("gene_id") == gene_id)["transcript_id"])

    # log.info(f"Canonical transcript: {canonical}")
    # print(appris)

    # if appris.filter(pl.col("transcript_id") != canonical).shape[0]:
    #     log.warning(f"Canonical transcript {canonical} not found in APPRIS.")
    #     canonical = appris[0, "transcript_id"]


# fmt: off
@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--gene", "-g", type=str)
@click.option("--canonical", "mode", flag_value="canonical", default=True, help="Outputs canonical transcript only")
@click.option("--gencode"  , "mode", flag_value="gencode", help="Outputs all transcripts from GENCODE basic")
@click.option("--ensembl"  , "mode", flag_value="ensembl", help="Outputs all transcripts from Ensembl")
@click.option("--appris"   , "mode", flag_value="appris" , help="Outputs all principal transcripts from APPRIS (dominant coding transcripts)")
# fmt: on
def gettranscript(
    path: Path,
    gene: str,
    mode: Literal["gencode", "ensembl", "canonical", "appris", "apprisalt"] = "canonical",
):
    """Get transcript ID from gene name or gene ID"""
    res = _gettranscript(path, gene, mode)
    res = res if isinstance(res, list) else [res]
    log.info(f"Transcript ID(s): {', '.join(res)}")
