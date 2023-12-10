# %%
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import cycle, islice
from pathlib import Path

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


class ProbeSet(BaseModel):
    name: str
    species: str
    codebook: str
    bcidx: int


# %%


# %%


# %%
path = Path("starwork2")


def pad(s: str, target: int = 101):
    if len(s) > target + 2:
        raise ValueError("Too long")
    if len(s) > target:
        return s
    return s + "ACTCACCTAC"[: target - len(s)]


def backfill(seq: str, target: int = 149):
    return (
        "TTCCACTAACTCACATGTCATGCATTTCTTCTTACCCTGAGCAGATCAGTAGTCTTTACATGCTCGTAGTACCGTAAGCCAGATAC"[
            : max(0, target - len(seq))
        ]
        + seq
    )


def until_first_g(seq: str, target: str = "G"):
    r, target = rc(seq.upper()), target.upper()
    res, truncated = r[:6], r[6:]
    res += "" if res[-1] == target else truncated[: truncated.index(target) + 1]
    if len(res) > 12:
        raise ValueError("No G found")
    assert res[-1] == target
    return res


def generate_head_splint(padlock: str, rand: np.random.Generator) -> str:
    # Minimize the amount of secondary structures.
    # Must start with C.
    test = "TAA" + rc(padlock)
    # base_hp = hp(test, "dna")
    out = []
    for _ in range(10):
        cand = "".join(rand.choice(["A", "T", "C"], p=[0.125, 0.125, 0.75], size=3))
        if "CCCCC" in cand:
            continue
        out.append((cand, hp(cand + test, "dna")))

    cand, hp_ = min(out, key=lambda x: x[1])
    assert len(cand) == 3
    return cand


def run(probeset: ProbeSet, n: int = 24):
    idx = probeset.bcidx
    rand = np.random.default_rng(0)
    codebook = json.loads((path / probeset.codebook).read_text())
    logger.info(f"Loaded {probeset.codebook} with {len(codebook)} genes.")

    hsh = hash_codebook(codebook)
    tss = list(codebook)
    # gene_to_ts = {"-".join(ts.split("-")[:-1]): ts for ts in tss}
    # ts_to_gene = {v: k for k, v in gene_to_ts.items()}

    dfs = pl.concat(
        [
            pl.read_parquet(f"starwork2/output/{ts}_final_BamHIKpnI_{hsh}.parquet")
            # .sample(shuffle=True, seed=4, fraction=1)
            .sort(["priority", "hp"])[:n]
            for ts in tss
        ]
    )

    # First must be C for KpnI.
    # Last must be G for BamHI.
    logger.info("Generating splint header.")
    res = dfs.with_columns(
        rotated=(
            # head
            hfs[idx * 2 + 1, "header"][-3:]
            + pl.col("padlock").apply(lambda x: generate_head_splint(x, rand)).str.to_lowercase()
            + "tc"
            + pl.col("seq").apply(rc)
        ).apply(pad)
        + rc(hfs[idx * 2, "footer"][:3])
        + "".join(rand.choice(["A", "T", "C"], p=[0.25, 0.25, 0.5], size=2))  # to generate some diversity
        + hfs[idx * 2 + 1, "footer"][:3]
    )

    it = cycle("ATAAT")

    def splint_pad(seq: str, target: int = 47):
        if len(seq) > target:
            return seq
        return "".join(islice(it, target - len(seq))) + seq

    res = (
        # res.with_columns(
        #     splint_end=pl.col("rotated").apply(until_first_g).str.to_lowercase(),
        #     # mismatch
        # )
        res.with_columns(
            splint_=pl.col("splint").apply(rc)
            + "TC"
            + pl.col("rotated").str.slice(0, 6).apply(rc)
            + pl.col("rotated").str.slice(-8, 8).apply(rc)
        ).with_columns(splint_=pl.col("splint_").apply(splint_pad))
    )

    assert (res["splint_"].apply(lambda x: BamHI.search(Seq.Seq(x))).list.lengths() == 0).all()
    assert (res["splint_"].apply(lambda x: KpnI.search(Seq.Seq(x))).list.lengths() == 0).all()
    assert (res["rotated"].apply(lambda x: BamHI.search(Seq.Seq(x))).list.lengths() == 0).all()
    assert (res["rotated"].apply(lambda x: KpnI.search(Seq.Seq(x))).list.lengths() == 0).all()

    def double_digest(s: str) -> str:
        return BamHI.catalyze(KpnI.catalyze(Seq.Seq(s))[1])[0].__str__()

    for s, r in zip(res["splint_"], res["rotated"]):
        assert test_splint_padlock(s, r, lengths=(6, 8)), (s, r)

    out = res.with_columns(
        # restriction scar already accounted for
        splintcons=hfs[idx * 2, "header"] + pl.col("splint_") + hfs[idx * 2, "footer"][3:],
        padlockcons=hfs[idx * 2 + 1, "header"][:-3].lower()
        + pl.col("rotated")
        + hfs[idx * 2 + 1, "footer"][3:],
    ).with_columns(splintcons=pl.col("splintcons").apply(backfill))

    for s, r in zip(out["splintcons"], out["padlockcons"]):
        assert test_splint_padlock(*map(double_digest, (s, r)), lengths=(6, 8)), (s, r)

    assert (out["padlockcons"].str.lengths().is_between(139, 150)).all()

    (gen_path := path / "generated").mkdir(exist_ok=True)
    out.write_parquet(gen_path / (probeset.name + ".parquet"))
    logger.info(f"{len(out)} probe pairs written to {gen_path/ (probeset.name + '.parquet')}")

    # Path("starwork/genestar_out.txt").write_text("\n".join([*out["padlockcons"], *out["splintcons"]]))

    (gen_path / (probeset.name + "_pad.fasta")).write_text(
        gen_fasta(out["padlockcons"], names=range(len(out))).getvalue()
    )
    (gen_path / (probeset.name + "_splint.fasta")).write_text(
        gen_fasta(out["splintcons"], names=range(len(out))).getvalue()
    )
    return out


# %%


def main():
    mfs = TypeAdapter(list[ProbeSet]).validate_json(Path("starwork2/manifest.json").read_text())
    print(sum(map(lambda x: len(run(x)), mfs)) * 2)

    species_mapping = {"mouse": "mus musculus", "human": "homo sapiens"}

    with ThreadPoolExecutor(4) as exc:
        for m in mfs:
            for ps in ["splint", "pad"]:
                exc.submit(
                    subprocess.run,
                    f'RepeatMasker -pa 16 -norna -s -no_is -species "{species_mapping[m.species]}" {path}/generated/{m.name}_{ps}.fasta',
                    shell=True,
                    check=True,
                )

    superout = []
    for m in mfs:
        out = []
        for s, p in zip(
            pyfastx.Fastx((path / "generated" / f"{m.name}_splint.fasta.masked").as_posix()),
            pyfastx.Fastx((path / "generated" / f"{m.name}_pad.fasta.masked").as_posix()),
        ):
            if "N" not in s[1] and "N" not in p[1]:
                out.append(s[1])
                out.append(p[1])
        Path(path / "generated" / f"{m.name}_final.txt").write_text("\n".join(out))
        superout.extend(out)

    # Path(path / "generated" / "all_final.txt").write_text("\n".join(superout))


if __name__ == "__main__":
    main()


# t7 = "TAATACGACTCACTATAGGG"
# assert out["padlockcons"].str.contains(rc(t7)[:5]).all()

# %%


# for name in ["genestarpad.fasta", "genestarsplint.fasta"]:

# cons = dfs.with_columns(constructed=header + pl.col("seq") + footer)
# cons = cons.with_columns(constructed=pl.col("constructed").apply(backfill))
# # %%
# import pyfastx


# # %%

# Path("starwork/genestar_out.txt").write_text("\n".join(out))


# %%

# %%
