import json
import re
from pathlib import Path

import rich_click as click
from loguru import logger as log

from fishtools.ext.external_data import Dataset
from fishtools.io.pretty_print import jprint


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("genes", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
def chkgenes(path: Path, genes: Path):
    """Validate/check that gene names are canonical in Ensembl"""
    from fishtools.ext.fix_gene_name import check_gene_names

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
