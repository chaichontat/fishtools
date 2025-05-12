import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
import requests
import rich_click as click
from loguru import logger

from fishtools.utils.pretty_print import jprint

from ..ext.dataset import Dataset, ReferenceDataset
from ..ext.external_data import get_ensembl


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
        logger.error(f"Found {len(out)} genes for {ts}: {out}")
    return list(out)[0]


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("genes", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
def chkgenes(path: Path, genes: Path):
    """Validate/check that gene names are canonical in Ensembl"""
    from ..ext.fix_gene_name import check_gene_names

    ds = ReferenceDataset(path)
    if not ds.ensembl:
        raise ValueError("Not a ReferenceDataset. Cannot check genes.")

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
        logger.critical(f"Non-unique genes found: {', '.join(gs)}.\n")
        genes.with_suffix(".unique.txt").write_text("\n".join(sorted(list(s))))
        logger.error(f"Unique genes written to {genes.with_suffix('.unique.txt')}.\n")
        return

    converted, mapping, no_fix_needed = check_gene_names(ds.ensembl, gs, species=ds.species)
    print(converted)
    if mapping:
        logger.info("Mappings:")
        jprint(mapping)
        logger.info(f"Mapping written to {genes.with_suffix('.mapping.json')}.")
        genes.with_suffix(".mapping.json").write_text(json.dumps(mapping))
        logger.info(f"Converted genes written to {genes.with_suffix('.converted.txt')}.")
        genes.with_suffix(".converted.txt").write_text("\n".join(sorted(converted)))
    elif not no_fix_needed:
        logger.warning("Some genes cannot be found.")
    else:
        logger.info(f"{len(s)} genes checked out. No changes needed")
        genes.with_suffix(".converted.txt").write_text("\n".join(sorted(converted)))


def get_transcripts(
    dataset: Dataset,
    genes: list[str],
    mode: Literal["gencode", "ensembl", "canonical", "appris", "apprisalt"] = "canonical",
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Get transcript ID from gene name or gene ID
    Returns:
        pl.DataFrame[[transcript_id, transcript_name, tag]]
        pl.DataFrame[[transcript_id, transcript_name, annotation, tag]] if appris
    """
    if not genes:
        raise ValueError("No genes provided")

    if not dataset.ensembl:
        raise ValueError("Not a ReferenceDataset. Cannot get transcripts.")

    df_genes = dataset.ensembl.filter(pl.col("gene_name").is_in(genes))[
        ["gene_name", "gene_id", "transcript_name", "transcript_id"]
    ]

    if dataset.appris is not None:
        df_genes = df_genes.join(
            dataset.appris.filter(pl.col("gene_id").is_in(df_genes["gene_id"]))[
                ["transcript_id", "annotation"]
            ],
            on="transcript_id",
            how="left",
        ).sort("transcript_name")

    to_return = ["gene_name", "gene_id", "transcript_id", "transcript_name", "tag"]

    match mode:
        case "canonical":
            with ThreadPoolExecutor(3) as exc:
                from functools import partial

                res = exc.map(partial(get_ensembl, output_path or "output/"), df_genes["gene_id"])
                canonical = [r["canonical_transcript"].split(".")[0] for r in res]

            res = dataset.ensembl.filter(pl.col("transcript_id").is_in(canonical))[to_return]
        case "gencode":
            res = dataset.data.filter(pl.col("gene_id").is_in(df_genes["gene_id"]))[to_return]
        case "ensembl":
            res = dataset.ensembl.filter(pl.col("gene_id").is_in(df_genes["gene_id"]))[to_return]
        case "appris":
            if dataset.appris is None:
                raise ValueError("No APPRIS data found.")
            appris = df_genes.filter(pl.col("annotation").is_not_null())
            # if len(principal := appris.filter(pl.col("annotation").str.contains("PRINCIPAL"))):
            #     logger.info("Principal transcripts: " + "\n".join(principal["transcript_id"]))
            res = appris.join(dataset.ensembl[["transcript_id", "tag"]], on="transcript_id", how="left")

            def handle_transcripts(group: pl.DataFrame):
                if len(group) > 1:
                    canonical = group.filter(pl.col("tag") == "Ensembl_canonical")
                    if len(canonical) == 1:
                        return canonical
                    principal = group.filter(pl.col("annotation").str.contains("PRINCIPAL"))
                    if len(principal) == 1:
                        return principal
                return group

            res = res.group_by("gene_name").map_groups(handle_transcripts)

        case "apprisalt":
            if dataset.appris is None:
                raise ValueError("No APPRIS data found.")
            appris = df_genes.filter(pl.col("annotation").is_not_null())
            res = appris.join(dataset.ensembl[["transcript_id", "tag"]], on="transcript_id", how="left")
        case _:  # type: ignore
            raise ValueError(f"Unknown mode: {mode}")

    # for gene, tss in sorted(res.group_by("gene_name"), key=lambda x: x[0]):
    #     if len(tss) > 1:
    #         print(
    #             f"Multiple transcripts found for {gene[0]}. See https://useast.ensembl.org/Mouse/Search/Results?q={gene[0]};site=ensembl;facet_species=Mouse"
    #         )
    #         print(f"Please pick one: {tss.with_row_index()}.")
    #         picked = input("Enter the index of the correct transcript: ")
    #         out.append(tss[int(picked)])
    #     else:
    #         out.append(tss)
    # try:
    #     out = pl.concat(out)
    # except ValueError:
    #     raise ValueError(f"No transcripts found for {genes}")
    # return out
    return res


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("genes", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["gencode", "ensembl", "canonical", "appris", "apprisalt"]),
    default="canonical",
)
def convert_to_transcripts(
    path: Path,
    genes: Path,
    mode: Literal["gencode", "ensembl", "canonical", "appris", "apprisalt"] = "canonical",
):
    """Validate/check that gene names are canonical in Ensembl"""
    ds = Dataset(path)
    del path
    gene_names = genes.read_text().splitlines()
    assert len(gene_names) == len(set(gene_names))
    res = get_transcripts(ds, gene_names, mode=mode)

    res = res.group_by("gene_name", maintain_order=True).agg(pl.all().first())

    genes.with_suffix(".tss.txt").write_text("\n".join(sorted(res["transcript_name"])))
    assert len(res["transcript_name"]) == len({
        x["transcript_name"].split("-")[0] for x in res.iter_rows(named=True)
    })
    logger.info(f"Written to {genes.with_suffix('.tss.txt')}.")


# fmt: off
@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.option("--gene", type=str)
@click.option("--genefile", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--canonical", "mode", flag_value="canonical", default=True, help="Outputs canonical transcript only")
@click.option("--gencode"  , "mode", flag_value="gencode", help="Outputs all transcripts from GENCODE basic")
@click.option("--ensembl"  , "mode", flag_value="ensembl", help="Outputs all transcripts from Ensembl")
@click.option("--appris"   , "mode", flag_value="appris" , help="Outputs all principal transcripts from APPRIS (dominant coding transcripts)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
# fmt: on
def transcripts(
    path: Path,
    gene: str | None = None,
    genefile: Path | None = None,
    mode: Literal["gencode", "ensembl", "canonical", "appris", "apprisalt"] = "canonical",
    verbose: bool = False,
):
    """Get transcript ID from gene name or gene ID"""

    if genefile:
        genes = genefile.read_text().splitlines()
    elif gene:
        genes = [gene]
    else:
        raise ValueError("No gene provided")
    res = get_transcripts(Dataset(path), genes, mode)
    if verbose:
        click.echo(res)
    else:
        click.echo("\n".join(sorted(res["transcript_name"].to_list())))
