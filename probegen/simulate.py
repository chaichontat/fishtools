# %%
from collections import deque
from pathlib import Path
import json
import polars as pl
from Bio import Restriction, Seq

from fishtools import Dataset, SAMFrame, gen_fasta, rc
from fishtools.mkprobes.starmap.starmap import test_splint_padlock
from fishtools.mkprobes.utils.sequtils import is_subsequence
from fishtools.utils.pretty_print import printc

hfs = pl.read_csv("data/headerfooter.csv")
t7promoter = "TAATACGACTCACTATAGGG"
bits = pl.read_csv("data/readout_ref_filtered.csv")
bits = dict(zip(bits["seq"], bits["id"]))

manifest = json.loads(Path("starwork5/_manifest.json").read_text())
idx = 1

species = manifest[idx]["species"]
ds = Dataset(f"data/{species}")
ss = Path(f"starwork5/generated/{manifest[idx]['name']}_final.txt").read_text().splitlines()
assert manifest[idx]["bcidx"] == idx


def t7(seq: str):
    """Find the sequence of the RNA generated by the T7 promoter."""
    try:
        idx = seq.index(t7promoter)
        return seq[idx + len(t7promoter) - 3 :]  # First base is G.
    except ValueError:
        ...

    idx = rc(seq).index(t7promoter)
    return rc(seq)[idx + len(t7promoter) - 3 :]


def rt(seq: str, primer: str):
    """Reverse transcribe the RNA sequence given a primer,"""
    try:
        res = rc(seq.upper()).index(primer.upper())
    except ValueError:
        raise ValueError("Primer not found.")
    return rc(seq)[res:]


def double_digest(s: str) -> str:
    return Restriction.BamHI.catalyze(Restriction.KpnI.catalyze(Seq.Seq(s))[1])[0].__str__()


def anneal(seq: str, primer: str):
    # Returns span of annealed primer.
    return is_subsequence(rc(primer).lower())(seq.lower())


# %%
def process(seq: str, rt_primer: str, anneal_primer: tuple[str, str]):
    t7ed = t7(seq + "TAGTGAGTCGTATTA")  # complete T7 promoter with PCR primers
    print(rt_primer)
    rted = rt(t7ed, rt_primer)

    # Confirm that the annealed sequences can be cut.
    frag1 = anneal(rted, anneal_primer[0])
    frag2 = anneal(rted, anneal_primer[1])
    assert frag1
    assert frag2

    return double_digest(rted)


i = 60
sp = process(
    ss[2 * i],
    hfs[2 * idx, "header"],
    (rc(hfs[2 * idx, "header"]), rc(hfs[2 * idx, "footer"][:-2])),
)
pad = process(
    ss[2 * i + 1],
    hfs[2 * idx + 1, "header"],
    ((rc(hfs[2 * idx + 1, "header"]), rc(hfs[2 * idx + 1, "footer"][:-2]))),
)

# pad = "GTAGACTACACACGGACACATTCATCTCTAACTCACATACACTAAAGATTGGTTC"
# sp = "GGCTTCCCATAGGTGTATATGCTAGTCTACGAACCA"

# %%
df = SAMFrame.from_bowtie(
    gen_fasta(
        # forward handle of splint, last 17 bases of splint are ligation handles
        # first 9 bases of pad are ligation handle
        [sp[3:-14], pad[6:27]],
        # [sp[3 : -(2 + 8 + 6)], pad[8:29]],
        names=["splint", "padlock"],
    ).getvalue(),
    f"data/{species}/txome",
    seed_length=12,
    threshold=15,
    n_return=200,
    fasta=True,
    no_forward=True,
)
# %%


# %%
sps = set(df.filter(pl.col("name") == "splint").sort("match_consec")["transcript"])
pads = set(df.filter(pl.col("name") == "padlock").sort("match_consec")["transcript"])
filtered = df.filter(pl.col("transcript").is_in(sps & pads)).sort(["transcript", "name"])
assert filtered.select((pl.col("flag") & 16 != 0).all())[0, "flag"]  # reverse strand
print(filtered[["name", "flag", "transcript", "pos"]])

idxs = (filtered[0, "pos"] + filtered[0, "match_consec"], filtered[1, "pos"])
gap = idxs[1] - idxs[0]
print(gap)
assert 0 <= gap <= 3  # distance between end of splint and start of padlock less than 3
# assert test_splint_padlock(sp[-14:], pad, lengths=(6, 8))
assert test_splint_padlock(sp, pad, lengths=(6, 6))


# %%

transcript = ds.ensembl.get_seq(filtered[0, "transcript"])

# %%

start = filtered[0, "pos"]
startpad = filtered[1, "pos"]


def show_starmap(
    transcript: str,
    pad: str,
    splint: str,
    gap: int = 0,
    pad_tail: int = 6,
    pad_hang: int = 6,
    splint_gap: int = 2,
):
    overhang = pad_tail + pad_hang + splint_gap
    space = " " * (len(splint) - overhang - 1)
    # extra 3 bases from padlock end
    printc(space + " " * (gap + 1) + "/---")
    for i in range(3):
        printc(space + ("↑" if i == 2 else " ") + (" " * gap) + pad[-(pad_tail + 3 - i)])

    # splint-padlock
    for i in range(pad_tail):
        a, b = splint[-(i + 1)], pad[-(pad_tail - i)]
        printc(space + a + ("-" if rc(a) == b else " ") * gap + b)

    printc(space + "|" + " " * gap + "-")

    for i in range(pad_hang + splint_gap):
        a, b = splint[-(i + pad_tail + 1)], pad[i]
        printc(space + a + ("-" if rc(a) == b else " ") * gap + b)

    combi = splint[:-overhang] + " " * gap + pad[pad_hang + splint_gap : pad_hang + 25] + "---"
    template = transcript[idxs[0] - 27 : idxs[1] + len(splint) - overhang - 1][::-1]

    # alignment
    printc(combi)
    print("".join("|" if rc(a) == b else " " for a, b in zip(combi, template)))
    print(template)


# show_starmap(transcript, df[10, "cons_pad"], df[10, "cons_splint"], pad_hang=6, pad_tail=6, gap=0)
show_starmap(transcript, pad, sp, gap=1)
# assert len(sp) == 45
# assert 91 <= len(pad) <= 107

# %%


def visualize_padlock(splint: str, pad: str):
    printc("5-" + splint[-12:-6] + "  " + splint[-6:] + "-3")
    printc("/-" + pad[:6][::-1] + "53" + pad[-6:][::-1] + "-\\")


visualize_padlock(sp, pad)


# %%
def rotate(s: str, r: int):
    return s[r:] + s[:r]


rcaed = rc(pad)  # reverse complement as a result of RCA
assert len(bits_ := [bits[bit] for bit in bits if rc(bit) in rcaed]) == 3
print(bits_)

# %%
# %%


# %%
df = pl.DataFrame({"seq": ss, "is_pad": [bool(i & 1) for i in range(len(ss))]})
pad = df.filter(pl.col("is_pad"))
# %%
print(pad["seq"][0].index(rc("TAATGGTCTCCTGGC")))

p = pad.select(pl.col("seq").str.slice(109 - 6, 20).str.extract_all(r"\w").list.to_struct()).unnest("seq")

# %%
mapping = {
    "['A', 'C', 'T']": "H",
    "['A', 'G', 'T']": "D",
    "['A', 'C', 'G']": "V",
    "['C', 'G', 'T']": "B",
    "['A', 'C']": "M",
    "['A', 'G']": "R",
    "['A', 'T']": "W",
    "['C', 'G']": "S",
    "['C', 'T']": "Y",
    "['G', 'T']": "K",
    "['A']": "A",
    "['C']": "C",
    "['G']": "G",
    "['T']": "T",
}

rc("".join([mapping[str(sorted(p[c].unique().to_list()))] for c in p.columns]))
# %%