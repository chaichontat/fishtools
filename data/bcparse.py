# %%
import io
from itertools import chain
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import primer3
import pyfastx
from Bio.Restriction import BamHI, KpnI
from Bio.Seq import Seq

from fishtools import hp, rc, tm

if not Path("data/bc_elledge.parquet").exists():
    f = pyfastx.Fasta("data/bc25mer.240k.fasta")
    df = (
        pl.DataFrame({"seq": [s.seq for s in f]})
        .with_row_count("id")
        .with_columns(
            hp=pl.col("seq").apply(lambda x: hp(x, "q5")), tm=pl.col("seq").apply(lambda x: tm(x, "q5"))
        )
    )
    df.write_parquet("data/bc_elledge.parquet")


def make_header(seq: str, extra: str):
    assert len(extra) == 2
    return seq + "GGTACC" + extra


def make_footer(seq: str, extra: str):
    # BamHI + seq + optimal start + T7 tail
    return extra + "GGATCC" + seq + "ACTCTCCCTA"


bcs = pl.read_parquet("data/bc_elledge.parquet")


def gen_bc(
    bcs: pl.DataFrame,
    rng: slice,
    f: Callable[[str, str], str],
    length: int,
    n_to_gen: int = 200,
):
    ok: list[str] = []
    built: list[str] = []
    for row in bcs[rng].iter_rows(named=True):
        if len(ok) > n_to_gen:
            break
        if len(BamHI.catalyze(Seq(row["seq"]))) > 1 or len(KpnI.catalyze(Seq(row["seq"]))) > 1:
            continue
        h = f(c := row["seq"][:length], row["seq"][-2:])
        if not (68 <= tm(h, "q5") <= 70) or primer3.calc_homodimer_tm(h) > 45 or hp(h, "q5") > 40:
            continue
        ok.append(c)
        built.append(h)
    return ok, built


headers, hb = gen_bc(bcs, slice(0, 10000), make_header, 12)
footers, fb = gen_bc(bcs, slice(10000, 20000), make_footer, 9)

res = pl.DataFrame(
    {
        "header": hb,
        "footer": fb,
        "header_ori": headers,
        "footer_ori": footers,
    }
).with_row_count("id")

for row in res.iter_rows(named=True):
    assert primer3.calc_heterodimer_tm(row["header"], rc(row["footer"])) < 50

if (p := Path("data/headerfooter.csv")).exists():
    res.write_csv(x := io.BytesIO())
    assert p.read_bytes() == x.getvalue()

res.write_csv("data/headerfooter.csv")

# %%


def gen_idt(name: str, seq: str, scale: str):
    return f"{name}\t{seq}\t\t{scale}\tSTD"


t7 = "GAATTTAATACGACTCACTATAGGG"


def gen_primer_set(i: int):
    def oneset(name: str, j: int):
        assert t7 in (footer := t7[:-5] + rc(res[j, "footer"]))
        return [
            gen_idt(f"{name}-{i}-Header", res[j, "header"], "25nm"),
            gen_idt(f"{name}-{i}-Footer", footer, "25nm"),
            gen_idt(f"{name}-{i}-Cleave1", rc(res[j, "header"]), "25nm"),
            gen_idt(f"{name}-{i}-Cleave2", rc(res[j, "footer"][:-3]), "25nm"),
        ]

    return oneset("Spl", 2 * i) + oneset("Pad", 2 * i + 1)


print("\n".join(chain.from_iterable([gen_primer_set(i) for i in range(7)])))
print(gen_idt("T7Footer-Shared-R", "GAATTTAATACGACTCACTATAGGGAGAGT", "25nm"))

# %%
