import json
import re
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
import requests
import rich_click as click
from loguru import logger as log

from fishtools.utils.pretty_print import jprint

from ..ext.external_data import Dataset, get_ensembl


def find_outdated_ts(ts: str) -> tuple[Annotated[str, "gene_name"], Annotated[str, "gene_id"]]:
    if not ts.startswith("ENST"):
        raise ValueError(f"{ts} is not a human Ensembl transcript ID")
    out = set()
    for x in requests.get(
        "https://dev-tark.ensembl.org/api/transcript/",
        params={
            "stable_id": ts,
            "source_name": "ensembl",
            "assembly_name": "GRCh38",
            "expand": "genes",
        },
    ).json()["results"]:
        for y in x["genes"]:
            if y["name"]:
                out.add((y["name"], y["stable_id"]))

    if len(out) != 1:
        log.error(f"Found {len(out)} genes for {ts}: {out}")
    return list(out)[0]


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("genes", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
def chkgenes(path: Path, genes: Path):
    """Validate/check that gene names are canonical in Ensembl"""
    from ..ext.fix_gene_name import check_gene_names

    ds = Dataset(path)
    del path
    gs: list[str] = re.split(r"[\s,]+", genes.read_text())
    if not gs:
        raise ValueError("No genes provided")

    gs = list(filter(lambda x: x, gs))
    for gene in gs:
        if not gene.isascii():
            raise ValueError(f"{gene} not ASCII.")

    if len(gs) != len(s := set(gs)):
        [gs.remove(x) for x in s]
        log.warning(f"Non-unique genes found: {', '.join(gs)}.\n")
        genes.with_suffix(".unique.txt").write_text("\n".join(sorted(list(s))))
        log.error(f"Unique genes written to {genes.with_suffix('.unique.txt')}.\n")
        log.critical("Aborting.")
        return

    converted, mapping, no_fixed_needed = check_gene_names(ds.ensembl, gs, species=ds.species)
    if mapping:
        log.info("Mappings:")
        jprint(mapping)
        log.info(f"Mapping written to {genes.with_suffix('.mapping.json')}.")
        genes.with_suffix(".mapping.json").write_text(json.dumps(mapping))
        log.info(f"Converted genes written to {genes.with_suffix('.converted.txt')}.")
        genes.with_suffix(".converted.txt").write_text("\n".join(sorted(converted)))
    elif not no_fixed_needed:
        log.warning("Some genes cannot be found.")
    else:
        log.info(f"{len(s)} genes checked out. No changes needed")
        genes.with_suffix(".converted.txt").write_text("\n".join(sorted(converted)))


def get_transcripts(
    dataset: Dataset,
    gene: str,
    mode: Literal["gencode", "ensembl", "canonical", "appris", "apprisalt"] = "canonical",
) -> pl.DataFrame:
    """Get transcript ID from gene name or gene ID
    Returns:
        pl.DataFrame[[transcript_id, transcript_name, tag]]
        pl.DataFrame[[transcript_id, transcript_name, annotation, tag]] if appris
    """
    gene_id = gene if gene.startswith("ENS") else dataset.ensembl.gene_to_eid(gene)
    del gene

    # log.info(f"Gene ID: {gene_id}")

    ensembl = dataset.ensembl.filter(pl.col("gene_id") == gene_id)[
        ["gene_name", "gene_id", "transcript_name", "transcript_id"]
    ]
    ensembl = ensembl.join(
        dataset.appris.filter(pl.col("gene_id") == gene_id)[["transcript_id", "annotation"]],
        on="transcript_id",
        how="left",
    ).sort("transcript_name")
    appris = ensembl.filter(pl.col("annotation").is_not_null())

    to_return = ["transcript_id", "transcript_name", "tag"]

    match mode:
        case "canonical":
            canonical = get_ensembl("output/", gene_id)["canonical_transcript"].split(".")[0]
            return dataset.ensembl.filter(pl.col("transcript_id") == canonical)[to_return]
        case "gencode":
            return dataset.gencode.filter(pl.col("gene_id") == gene_id)[to_return]
        case "ensembl":
            return dataset.ensembl.filter(pl.col("gene_id") == gene_id)[to_return]
        case "appris":
            # if len(principal := appris.filter(pl.col("annotation").str.contains("PRINCIPAL"))):
            #     log.info("Principal transcripts: " + "\n".join(principal["transcript_id"]))
            return appris.filter(pl.col("annotation").str.contains("PRINCIPAL")).join(
                dataset.ensembl[["transcript_id", "tag"]], on="transcript_id", how="left"
            )
        case "apprisalt":
            return appris.join(dataset.ensembl[["transcript_id", "tag"]], on="transcript_id", how="left")
        case _:  # type: ignore
            raise ValueError(f"Unknown mode: {mode}")

    # log.info(ensembl)
    # log.info(appris)


# fmt: off
@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("gene", type=str)
@click.option("--canonical", "mode", flag_value="canonical", default=True, help="Outputs canonical transcript only")
@click.option("--gencode"  , "mode", flag_value="gencode", help="Outputs all transcripts from GENCODE basic")
@click.option("--ensembl"  , "mode", flag_value="ensembl", help="Outputs all transcripts from Ensembl")
@click.option("--appris"   , "mode", flag_value="appris" , help="Outputs all principal transcripts from APPRIS (dominant coding transcripts)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
# fmt: on
def transcripts(
    path: Path,
    gene: str,
    mode: Literal["gencode", "ensembl", "canonical", "appris", "apprisalt"] = "canonical",
    verbose: bool = False,
):
    """Get transcript ID from gene name or gene ID"""
    res = get_transcripts(Dataset(path), gene, mode)
    if verbose:
        click.echo(res)
    else:
        click.echo("\n".join(res["transcript_id"].to_list()))
