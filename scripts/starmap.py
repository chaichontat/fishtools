# %%
from itertools import combinations, cycle
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from fishtools import rc

ros = pl.read_csv("data/readout_ref_filtered.csv")
# %%
pheaders = {"Malat1": "ACATTA", "Cux2": "AGGCTA", "Camk2a": "TCGATG"}

pfooters = {"Malat1": "AAGATA", "Cux2": "AAGATA", "Camk2a": "TCTGGA"}


def splintfooter(gene: str):
    return rc(pfooters[gene] + pheaders[gene])


gt_sf = {
    "Malat1": "TAATGTTATCTT",
    "Cux2": "TAGCCTTATCTT",
}

# assert all(v == splintfooter(k) for k, v in gt_sf.items())


def rotate(s: str, r: int):
    return s[r:] + s[:r]


# %%
def gen_interleave(ids: tuple[int, int], n: int):
    spacer = "AATTATTA"
    seqs = [ros.filter(pl.col("id") == i)["seq"][0] for i in ids]
    return spacer[:n].join(seqs)


# 68, 79 and 89 nt
gen_interleave((1, 2), 3)
# %%
binders = {
    "Malat1": """AGCATAGCAGTACACGCCTTCT
TGCAACGTGACCTCAAGGAT
CCACTTAAGTGTTCGAAGTCAAGT
TCAAGTGAGGTGACAAAAGGC""".splitlines(),
    "Cux2": """ACTCACGCTACCCTGAGACAGT
CTTGAGGTTGAGTTGGAAGGAG
GAGGGAGAGTTCGGTGGCCC
TGGCCATTTTCTCATGCCG""".splitlines(),
}

splints = {
    "Malat1": """CCCTGAAGGTGTCGTGCCA
ACCATGGTTACCTTGAAACCGA
CTGGCTCAAATCCTGATCTG
ATGCCTGACCCCACTCGTG""".splitlines(),
    "Cux2": """GCTTTGGCCGTGACAGCATG
ACCAGTTGATGACGGTGTTC
CGTGAGGTCTGGCGAGGAGATGA
CCAGCCGGTAGATTATACTGTTCAA""".splitlines(),
}

# %%


def gen_probes(gene: str, ids: tuple[int, ...], target: int = 79):
    for s, ids in zip(binders[gene], cycle(combinations(ids, 2))):
        length = len(s) + 40 + 12
        yield pheaders[gene] + s + gen_interleave(ids, target - length) + pfooters[gene]


malat1 = list(gen_probes("Malat1", (1, 2, 3)))
assert all(len(s) == 79 for s in malat1)

cux2 = list(gen_probes("Cux2", (3, 4, 5)))
assert all(len(s) == 79 for s in cux2)

# cycle(combinations((3, 4, 5), 2))


# %%
def gen_idt(name: str, seq: str, scale: str):
    return f"{name}\t{seq}\t\t{scale}\tSTD"


print("\n".join([gen_idt(f"STARPad_Malat1_{i}", s, "100nm") for i, s in enumerate(malat1, 1)]))

# %%
print("\n".join([gen_idt(f"STARPad_Cux2_{i}", s, "100nm") for i, s in enumerate(cux2, 1)]))

# %%
print(
    "\n".join(
        [
            gen_idt(f"STARSplint_Malat1_{i}", s + splintfooter("Malat1"), "25nm")
            for i, s in enumerate(splints["Malat1"], 1)
        ]
    )
)
print(
    "\n".join(
        [
            gen_idt(f"STARSplint_Cux2_{i}", s + splintfooter("Cux2"), "25nm")
            for i, s in enumerate(splints["Cux2"], 1)
        ]
    )
)
# %%
len("AACTACATACTCCCTACCTCTTTAACTACATACTCCCTACCTCTTT")


# %%


bit7 = "TATTCACCTTACAAACCCTC"  # need 6th to be a
bits = {i: ros.filter(pl.col("id") == i)["seq"][0] for i in range(1, 31)}
bit8 = ros.filter(pl.col("id") == 8)["seq"][0]
bit9 = ros.filter(pl.col("id") == 9)["seq"][0]
actb = """GGGATAGCACAGCCTGGATAGCAA
AGCGCGGCGATATCATCATC
ACCATCACGCCCTGGTGCC
CTGATTGGCCCCGCGCCGCT
CGGCAAAGGCGAGGCTCTGT
ggcgtacaggtctttgcgg""".splitlines()


# %%
def geneusplint(s: str):
    assert len(s) == 12
    k = list(s)
    k.insert(-2, "*")
    k.insert(-1, "*")
    s = "".join(k)

    return f"/5AmMC6/{'AAAAT'*10}{s}/3InvdT/"


# %%


# %%
def test_splint_padlock(splint: str, padlock: str):
    if not len(splint) & 1 == 0:
        raise ValueError("Splint must be even length")
    mid = len(splint) // 2
    return rc(splint[:mid]) == padlock[:mid] and rc(splint[mid:]) == padlock[-mid:]


# %%

# %%
from fishtools import tm


def split(seq: str, target_tm: float) -> tuple[str, str] | None:
    first, last = "", ""
    i = 0
    for i in range(18, len(seq)):
        if tm(seq[:i], "hybrid") - 5 > target_tm:
            first = seq[:i]
            break

    for offset in reversed(range(2)):
        for j in range(18, 30):
            if i + offset + j >= len(seq):
                break

            if tm(seq[i + offset : i + offset + j], "hybrid") - 5 > target_tm:
                last = seq[i + offset : i + offset + j]
                return first, last
    return "", ""
    # raise Exception("Could not find split")


gene = "EEF2-201"
df = (
    pl.read_parquet(f"output/{gene}_screened_ol-2.parquet")
    .with_columns(splited=pl.col("seq").apply(lambda s: split(rc(s), 60)))
    .filter(pl.col("splited").list.first().str.lengths() > 0)
    .sample(fraction=1, shuffle=True, seed=0)
    .sort(["priority", "oks"], descending=[False, True])
)
# splited = [split(rc(seq), 60) for seq in df["seq"]]
# splited = [s for s in splited if s and 0 < len(s[0]) < 30]

eusp = "GTAAGGTGAATA"


fulleusp = geneusplint(eusp)


def generate_pair(bs: list[int], splited: list[tuple[str, str]]):
    footer = "CA" + rc(bits[bs[0]][-11:-1])
    top2a_primer = [s[0] + footer for s in splited]
    top2a_pad = [
        "CCTTAC"
        + bits[bs[0]].lower()
        + "TA"
        + s[1]
        + bits[bs[1]].lower()
        + "TATTAAAT"[: 79 - len(s[1]) - 54]
        + "TATTCA"
        for s in splited
    ]
    assert all(test_splint_padlock(eusp, s) for s in top2a_pad)
    assert all(0 < s.index(rc(footer[2:]).lower()) < len(s) // 2 for s in top2a_pad)
    return top2a_primer, top2a_pad


# eusp = rc(rotate(top2a_pad[0], -6))[-12:]

from fishtools import hp


# %%
def test_and_filter(primer: list[str], pad: list[str], size: int = 79):
    hps = [max(hp(s[:60], "dna"), hp(s[-60:], "dna")) for s in pad]
    filtered = [(pri, pa) for pri, pa, h in zip(primer, pad, hps) if h < 60 and len(pa) == size]
    return filtered


# %%
top2as = []
top2as.extend(
    [gen_idt(f"TEMPOPrimer-{gene.split('-')[0]}-{i}", pri, "25nm") for i, (pri, _) in enumerate(filtered, 1)]
)
top2as.extend(
    [gen_idt(f"TEMPOPad-{gene.split('-')[0]}-{i}", pa, "100nm") for i, (_, pa) in enumerate(filtered, 1)]
)
top2as.append(gen_idt("TEMPOSplint-07", fulleusp, "100nm"))
# %%
print("\n".join(top2as))
# %%
r18sprimer = """CATGGCCTCAGTTCCGAAAACC
GCCGCATCGCCAGTCG
GACATCTAAGGGCATCACAGAC
ACCTTGTTACGACTTTTACTTCCTC""".splitlines()

r18spad = """ACAAAATAGAACCGCGGTCCTATTC
CATCGTTTATGGTCGGAACTACGAC
TGTTATTGCTCAATCTCGGGTGGC
AGATAGTCAAGTTCGACCGTCTTCT""".splitlines()


# %%
def gen_starpad(bis: list[int], splits: list[tuple[str, str]]):
    combi = cycle(combinations(bis, 2))
    # footer = "TA" + rc(roted[:6]) + rc(roted[-6:])

    def inner(s: tuple[str, str]):
        a, b = next(combi)
        out_pad = rotate(
            bits[a].lower() + "ta" + s[1] + bits[b].lower() + "TATTCAAT"[: 67 - len(s[1]) - 42], 14
        )
        splint = s[0] + (footer := "ta" + rc(out_pad[:6]) + rc(out_pad[-6:]))
        assert test_splint_padlock(footer[2:], out_pad), out_pad
        return splint, out_pad

    res = [inner(s) for s in splits]

    return [s[0] for s in res], [s[1] for s in res]


tempo = test_and_filter(*generate_pair([9, 17], df["splited"][::2].to_list()), size=79)[:5]
star = test_and_filter(*gen_starpad([10, 18], df["splited"][::2].to_list()), size=67)[:5]


rpsall = test_and_filter(*generate_pair([11, 19], list(zip(r18sprimer, r18spad))), size=79)[:5]

# gen_pp_idt(camk2a[:5])
# %%


def gen_pp_idt(gene: str, pp: list[tuple[str, str]], mode: Literal["TEMPO", "STAR"] = "TEMPO"):
    return [
        *[gen_idt(f"{mode}Primer-{gene.split('-')[0]}-{i}", pri, "25nm") for i, (pri, _) in enumerate(pp, 1)],
        *[gen_idt(f"{mode}Pad-{gene.split('-')[0]}-{i}", pa, "100nm") for i, (_, pa) in enumerate(pp, 1)],
    ]


print(
    "\n".join(
        [
            *gen_pp_idt("Rps18", rpsall, mode="TEMPO"),
            *gen_pp_idt("Hpcal4Fix", star, mode="STAR"),
            *gen_pp_idt("Hpcal4", tempo, mode="TEMPO"),
        ]
    )
)
# %%
df = pl.read_csv("starmaplit.csv", separator=",").with_columns(
    gene=pl.col("Primers").str.split("_").list.first()
)
# %%
genes = """Cux2 # L2
Rorb  # L4
Plcxd2  # L5
Igfbp4  # L6a
Ctss # Microglia
Mal  # Oligo
Gfap  # Astrocytes
Pdgfra  # OPC
Nptx1  # CA3
Slc17a6  # Glu
Adora2a  # Inh
Drd1  # Dopamine
Neurod6  # hipp
Slc17a7  # Vglut2
Vtn  # vascular""".splitlines()
# Gad2  # Inh sm
# Sst  # Inh sm

genes = [g.split("#")[0].strip() for g in genes]
mhd = np.loadtxt("static/mhd4_16bit.csv", delimiter=",", dtype=bool)
cb = {g: np.where(m)[0] + 1 for g, m in zip(genes, mhd)}
cb["Gad2"] = np.array([29, 29])
cb["Sst"] = np.array([30, 30])
genes += ["Gad2", "Sst"]

df = df.filter(pl.col("gene").is_in(genes)).with_columns(
    Primerseq=pl.col("Primerseq").str.replace(r"............$", ""),
    Padlockseq=pl.col("Padlockseq").str.slice(14).apply(lambda s: s[:-29]),
)

# %%


# %%
names = []
res = []
idtout = []
for gene in cb.keys():
    ddf = df.filter(pl.col("gene") == gene)
    x = gen_starpad(cb[gene], list(zip(ddf["Primerseq"], ddf["Padlockseq"])))
    print(gene, x)
    assert len(x[0]) == 4
    res.extend(x)
    idtout.extend(gen_pp_idt(gene + "Fix", [(q, w) for q, w in zip(*x)], mode="STAR"))

Path("star.tsv").write_text("\n".join(idtout))

# %%
print(
    "\n".join(
        [
            gen_idt(
                f"RCACondenser-{i}",
                "TAAA".join([bits[i][:16]] * 2)[:-2] + f"*{bits[i][-2]}*{bits[i][-1]}/3InvdT/",
                "100nm",
            )
            for i in range(1, 11, 3)
        ]
    )
)
# %%
from fishtools.mkprobes.utils.sequtils import gen_plate

xdf = pl.read_csv("star.tsv", separator="\t", has_header=False).with_columns(
    is_pad=pl.col("column_1").str.contains("Pad")
)
# gen_plate(xdf.filter(pl.col("is_pad"))[:, 0], xdf.filter(pl.col("is_pad"))[:, 1], order="F").write_excel(
#     "star_pad.xlsx"
# )
# %%
import json

Path("starmaptestcb.json").write_text(json.dumps({k: v.tolist() for k, v in cb.items()}))
# %%
