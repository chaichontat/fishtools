from itertools import chain
from typing import Iterable, Literal, TypedDict, cast

import mygene
import polars as pl
from loguru import logger as log

from fishtools.mkprobes.ext.external_data import _ExternalData
from fishtools.utils.pretty_print import jprint

mg = mygene.MyGeneInfo()


class GeneDict(TypedDict):
    gene: list[str]
    symbol: str


class ResDict(TypedDict):
    out: list[dict[str, str | float]]
    dup: list[str]
    missing: list[str]


def find_aliases(gtf: _ExternalData, genes: Iterable[str], species: str = "mouse"):
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

    # symbol is from Uniprot, can conflict with ensembl.
    eids = list(chain.from_iterable(x["gene"] for x in out.values()))
    df = gtf.filter(pl.col("gene_id").is_in(eids))
    for v in out.values():
        _res = df.filter(pl.col("gene_id").is_in(v["gene"]))[0, "gene_name"]
        if not isinstance(_res, str):
            continue
        v["symbol"] = _res

    # Simple case of duplication: only keep ensembl name.
    for d, _ in res["dup"].copy():
        queries = [x for x in res["out"] if x["query"] == d and "ensembl" in x["query"]]
        if len(queries) == 1:
            out[d] = GeneDict(gene=[queries[0]["ensembl"]["gene"]], symbol=queries[0]["symbol"])
            res["out"].remove((d, _))

    return out, cast(ResDict, res)  # missing, dup


def manual_fix(res: ResDict, sel: dict[str, str]):
    dupes = {x[0] for x in res["dup"]}
    input("Manual fix: press enter to start. Enter the number of the correct gene name.")

    for line in dupes:
        choices = {i: x for i, x in enumerate((x for x in res["out"] if x["query"] == line), 1)}
        query = choices[1]["query"]
        symbols = [x["symbol"] for x in choices.values()]
        if len(set(symbols)) == len(symbols):
            # No duplicates
            try:
                matching = symbols.index(query)
            except ValueError:
                ...
            else:
                sel[line] = choices[matching + 1]["symbol"]
                continue

        # Manual fix
        print(f"\nTarget is {query}. Which one is correct?\n")
        jprint(choices)
        inp = input()
        if not inp:
            break
        try:
            n = int(inp)
        except ValueError:
            sel[line] = inp
        else:
            sel[line] = choices[n]["symbol"] if n != 0 else line  # 0 means keep original


def check_gene_names(gtf: _ExternalData, genes: list[str], species: Literal["mouse", "human"] = "mouse"):
    notfound = []
    ok: list[str] = []
    for gene in genes:
        try:
            gtf.gene_to_eid(gene)
            ok.append(gene)
        except ValueError:
            notfound.append(gene)
    converted, res = find_aliases(gtf, notfound, species=species)
    mapping = {k: v["symbol"] for k, v in converted.items()}
    no_fix_needed = True

    if len(res["dup"]):
        log.info(f"Duplicates found {res['dup']}. Please fix by selecting the correct index.")
        manual_fix(res, mapping)
    if len(res["missing"]):
        log.error(f"Cannot find {res['missing']}")
        no_fix_needed = False
        # raise ValueError(f"Duplicated aliases {res['dup']} or missing aliases {res['missing']}")

    return ok + [x["symbol"] for x in converted.values()], mapping, no_fix_needed
