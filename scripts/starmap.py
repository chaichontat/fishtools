# %%
import polars as pl

from fishtools import rc

ros = pl.read_csv("data/readout_ref_filtered.csv")
# %%
pheaders = {"Malat1": "ACATTA", "Cux2": "AGGCTA"}

pfooters = {
    "Malat1": "AAGATA",
    "Cux2": "AAGATA",
}


def splintfooter(gene: str):
    return rc(pfooters[gene] + pheaders[gene])


gt_sf = {
    "Malat1": "TAGCCTTATCTT",
    "Cux2": "TAGCCTTATCTT",
}

assert all(v == splintfooter(k) for k, v in gt_sf.items())


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
from itertools import combinations, cycle


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
def rotate(s: str, r: int):
    return s[r:] + s[:r]


bit7 = "TATTCACCTTACAAACCCTC"  # need 6th to be a
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

    return f"/5AzideN/{'AAAAT'*10}{s}/3InvdT/"


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


def split(seq: str, target_tm: float):
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
    raise ValueError("Could not find split")


# %%

df = pl.read_parquet("output/Top2a-201_screened_ol-2.parquet").sort("oks", descending=True)
# %%


actb_footer = "CT" + rc(bit7[-10:])
splited = [split(rc(seq), 60) for seq in df[1:7]["seq"]]
splited = [s for s in splited if len(s[0]) < 30]

top2a_primer = [s[0] + actb_footer for s in splited]
top2a_pad = [rotate(bit7 + "TA" + s[1] + bit7 + "TATTAAAT"[: 67 - len(s[1]) - 42], 6) for s in splited]

eusp = rc(rotate(top2a_pad[0], -6))[-12:]

fulleusp = geneusplint(eusp)
# %%
assert test_splint_padlock(eusp, top2a_pad[0])
assert all(test_splint_padlock(eusp, s) for s in top2a_pad)
assert all(0 < s.index(rc(actb_footer[2:])) < len(s) // 2 for s in top2a_pad)
# %%
top2a_primer
# %%
top2as = []
top2as.extend([gen_idt(f"TEMPOPrimer-Top2a-{i}", s, "25nm") for i, s in enumerate(top2a_primer, 1)])
top2as.extend([gen_idt(f"TEMPOPad-Top2a-{i}", s, "100nm") for i, s in enumerate(top2a_pad, 1)])
top2as.append(gen_idt("TEMPOSplint-07", fulleusp, "100nm"))
# %%
print("\n".join(top2as))
# %%
