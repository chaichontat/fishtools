from typing import Iterable, TypedDict, cast

import mygene
from loguru import logger as log
from oligocheck.boilerplate import jprint

from fishtools.ext.external_data import ExternalData

mg = mygene.MyGeneInfo()


class GeneDict(TypedDict):
    gene: list[str]
    symbol: str


class ResDict(TypedDict):
    out: list[dict[str, str | float]]
    dup: list[str]
    missing: list[str]


def find_aliases(genes: Iterable[str], species: str = "mouse"):
    res = mg.querymany(
        genes, scopes="symbol,alias", fields="symbol,ensembl.gene", species=species, returnall=True
    )
    out: dict[str, GeneDict] = {}
    for x in res["out"]:
        if "ensembl" not in x:
            continue
        if isinstance(x["ensembl"], dict):
            eid = [x["ensembl"]["gene"]]
        else:
            eid = [y["gene"] for y in x["ensembl"]]
        out[x["query"]] = GeneDict(gene=eid, symbol=x["symbol"])

    # Simple case of duplication: only keep ensembl name.
    for d, _ in res["dup"].copy():
        queries = [x for x in res["out"] if x["query"] == d and "ensembl" in x["query"]]
        if len(queries) == 1:
            out[d] = GeneDict(gene=[queries[0]["ensembl"]["gene"]], symbol=queries[0]["symbol"])
            res["out"].remove((d, _))

    return out, cast(ResDict, res)  # missing, dup


def manual_fix(res: ResDict, sel: dict[str, str]):
    dupes = {x[0] for x in res["dup"]}
    for line in dupes:
        if line in sel:
            continue
        jprint(choices := {i: x for i, x in enumerate((x for x in res["out"] if x["query"] == line), 1)})
        inp = input()
        if not inp:
            break
        try:
            n = int(inp)
        except ValueError:
            sel[line] = inp
        else:
            sel[line] = choices[n]["symbol"] if n != 0 else line  # 0 means keep original


def check_gene_names(gtf: ExternalData, genes: list[str]):
    notfound = []
    ok: list[str] = []
    for gene in genes:
        try:
            gtf.gene_to_eid(gene)
            ok.append(gene)
        except ValueError:
            notfound.append(gene)
    converted, res = find_aliases(notfound)
    mapping = {k: v["symbol"] for k, v in converted.items()}
    no_fix_needed = True

    if len(res["dup"]):
        log.info("Duplicates found. Please fix by selecting the correct index.")
        manual_fix(res, mapping)
    if len(res["missing"]):
        log.error(f"Cannot find {res['missing']}")
        no_fix_needed = False
        # raise ValueError(f"Duplicated aliases {res['dup']} or missing aliases {res['missing']}")

    return ok + [x["symbol"] for x in converted.values()], mapping, no_fix_needed
