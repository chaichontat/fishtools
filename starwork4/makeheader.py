# %%
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, cycle, islice
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import pyfastx
from Bio import Seq
from Bio.Restriction import BamHI, KpnI
from loguru import logger
from pydantic import BaseModel, TypeAdapter

from fishtools import gen_fasta, hp, rc
from fishtools.mkprobes.codebook.codebook import hash_codebook
from fishtools.mkprobes.starmap.starmap import test_splint_padlock

pl.Config.set_fmt_str_lengths(100)
hfs = pl.read_csv("data/headerfooter.csv")


# %%


def gen_idt(name: str, seq: str, scale: str):
    return f"{name}\t{seq}\t\t{scale}\tSTD"


t7 = "GAATTTAATACGACTCACTATAGGG"


def gen_primer_set(i: int):
    def oneset(name: str, j: int):
        assert t7 in (footer := t7[:-5] + rc(hfs[j, "footer"]))
        return [
            gen_idt(f"{name}-{i}-Header", hfs[j, "header"], "100nm"),
            # gen_idt(f"{name}-{i}-Footer", footer, "25nm"),
            gen_idt(f"{name}-{i}-Cleave1", rc(hfs[j, "header"]), "100nm"),
            gen_idt(f"{name}-{i}-Cleave2", rc(hfs[j, "footer"][:-3]), "100nm"),
        ]

    return oneset("Spl", 2 * i) + oneset("Pad", 2 * i + 1)


print("\n".join(chain.from_iterable([gen_primer_set(i) for i in range(2)])))


# def gen():
#     out = []

#     def run(idx: int, mode: Literal["spl", "pad"]):
#         gen_name = lambda x: "-".join([mode.capitalize(), x, str(idx)])

#         actual_idx = idx * 2 + (mode == "pad")

#         this = [
#             gen_idt(gen_name("Header"), hfs[actual_idx, "header"]),
#             gen_idt(gen_name("Footer"), "GAATTTAATACGACTCACTA" + rc(hfs[actual_idx, "footer"])),
#         ]

#     #

# %%
