# %%
"""
From the paper:

TATCTTTAGT*G*T*/3InvdT/
ACACTA AAGATA

Tri-probe_ACTB_primer_1	 TGGTACGGCCAGAGGCGTA     GT GCGTACAGTA
Tri-probe_ACTB_primer_2	 CGGAGCCGTTGTCGACGAC     GT GCGTACAGTA
Tri-probe_ACTB_primer_3	 TAGGAATCCTTCTGACCCATGC  GT GCGTACAGTA
Tri-probe_ACTB_primer_4	 AGGCAACTTTCGGAACGGCGCA  GT GCGTACAGTA
Tri-probe_ACTB_primer_5	 GGTGTGGACGGGCGGCGGA     GT GCGTACAGTA
Tri-probe_ACTB_primer_6	 gccgccagacagcactgtgt    GT GCGTACAGTA

Tri-probe_ACTB_pad_1	/5Phos/AAGATA ACA TACTGTACGC TA GGGATAGCACAGCCTGGATAGCAA ATTACGTACCCAT ACACTA
Tri-probe_ACTB_pad_2	/5Phos/AAGATA ACA TACTGTACGC TA AGCGCGGCGATATCATCATC     ATTACGTACCCAT ACACTA
Tri-probe_ACTB_pad_3	/5Phos/AAGATA ACA TACTGTACGC TA ACCATCACGCCCTGGTGCC      ATTACGTACCCAT ACACTA
Tri-probe_ACTB_pad_4	/5Phos/AAGATA ACA TACTGTACGC TA CTGATTGGCCCCGCGCCGCT     ATTACGTACCCAT ACACTA
Tri-probe_ACTB_pad_5	/5Phos/AAGATA ACA TACTGTACGC TA CGGCAAAGGCGAGGCTCTGT     ATTACGTACCCAT ACACTA
Tri-probe_ACTB_pad_6	/5Phos/AAGATA ACA TACTGTACGC TA ggcgtacaggtctttgcgg      ATTACGTACCCAT ACACTA

From https://academic.oup.com/nar/article/46/2/538/4716935, we need 67+(10*n) bp for padlock.
"""

import json
import types
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import polars as pl

from fishtools import rc
from fishtools.mkprobes.starmap.simple import process, run_pipeline
from fishtools.mkprobes.utils.sequtils import gen_idt

P, T = ParamSpec("P"), TypeVar("T")


def imports():
    for name, val in globals().items():
        # module imports
        if isinstance(val, types.ModuleType):
            yield name, val
        # functions / callables
        if hasattr(val, "__call__"):
            yield name, val


readouts = pl.read_csv("data/readout_ref_filtered.csv")


species = "mouse"
cwd = Path("tempotest")
codebook_path = Path(f"{cwd}/tempo.json").resolve()
codebook = json.loads(codebook_path.read_text())
genes = {name.split("-")[0]: v for name, v in codebook.items()}

run_pipeline(codebook_path, cwd, species)


# %% [markdown]
# Using bit 10 as a reference: ACCCAACACTCATAACATCC
# Since this is going to be used as a part of the universal splint, the homology length cannot be too long.
#
# Hence from above:
# Tri-probe_ACTB_pad_1	/5Phos/AAGATA ACA TACTGTACGC TA GGGATAGCACAGCCTGGATAGCAA ATTACGTACCCAT ACACTA
#
# we will keep the 5' part of the padlock and put the bit right after the gene sequence with another C at the end to "disguise".
#
#                       /5Phos/AAGATA ACA TACTGTACGC TA GGGATAGCACAGCCTGGATAGCAA ACCCAACACTCATAACATCC ACACTA


# %%
df = process(codebook, cwd, n=16)


# %%
def gen_tempo(splint: str, padlock: str, bit: int):
    """Assume not RC'd at this point."""
    PADDING = "ATCATAACAAATA"
    PAD_HEAD = "AAGATA ACA TACTGTACGC TA"
    PAD_TAIL = "ACACTA"

    readout = readouts[bit, "seq"]
    print(len(padlock))
    TARGET_LEN = 77 - (PAD_HEAD + PAD_TAIL).replace(" ", "").__len__() - len(readout)
    out_sp = rc(splint).lower() + "GTGCGTACAGTA"
    if len(padlock) > TARGET_LEN:
        raise ValueError("Pad too long")
    out_pad = (
        PAD_HEAD + rc(padlock).lower() + PADDING[: TARGET_LEN - len(padlock)] + readout.lower() + PAD_TAIL
    )
    out_pad = out_pad.replace(" ", "")

    assert len(out_pad) == 77, len(out_pad)
    return out_sp, "/5Phos/" + out_pad


@noglobal
def final_print(genes: dict[str, list[int]], gene: str, _dfg: pl.DataFrame):
    spls, pads = [], []
    for i in range(6):
        spl, pad = gen_tempo(_dfg[i, "splint"], _dfg[i, "padlock"], genes[gene][0])
        print(gene)
        print(gen_idt(f"SplTEMPO-{gene}-{i + 1}", spl, "25nm"))
        print(pad)
        spls.append(spl)
        pads.append(pad)
    return spls, pads


spls = []
pads = []
for (gene,), _dfg in df.group_by(pl.col("name").str.split("_").list.get(0)):
    _spls, _pads = final_print(genes, gene, _dfg)
    spls.extend(_spls)
    pads.extend(_pads)
    # print(gen_idt(f"PadTEMPO-{gene}-{i + 1}", pad, "100nm"))


# %% [markdown]
# TEMPO-Splint	/5AmMC6/AAAAAAATAAAAAAATAAAAAAATAAAAAAATAAAAAAATAAAAAATATCTTTAGT*G*T*/3InvdT/		100nm	HPLC
# print(gen_idt(f"PadTEMPO-{gene}-{i + 1}", pad, "100nm"))

print("\n".join(spls))
# %%
print("\n".join(spls + pads))
# %%
