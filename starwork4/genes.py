# %%
import json
from itertools import chain
from pathlib import Path
from subprocess import run
from typing import Literal

import click
import polars as pl
from loguru import logger

from fishtools.mkprobes.codebook.codebook import CodebookPicker
from fishtools.mkprobes.ext.external_data import Dataset
from fishtools.mkprobes.genes.chkgenes import get_transcripts
from fishtools.mkprobes.starmap.coding import gen_codebook, order


@click.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.argument("path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option("--existing", "-e", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--single", is_flag=True)
@logger.catch
def main(data: Path, path: Path, existing: Path, single: bool = False):
    # if "converted" in path.name:
    #     raise ValueError("Path should be to the original file")
    existing_cb = json.loads(existing.read_text()) if existing else None

    if not path.name.endswith(".tss.txt"):
        path.with_suffix(".converted.txt").unlink(missing_ok=True)

        run(f"mkprobes chkgenes {data} {path}", shell=True, check=True)
        ds = Dataset(data)
        try:
            corrected = path.with_suffix(".converted.txt").read_text().splitlines()
        except FileNotFoundError:
            corrected = path.read_text().splitlines()

        corrected = set(corrected)

        def parse(df: pl.DataFrame) -> pl.DataFrame:
            if len(res := df.filter(pl.col("tag") == "Ensembl_canonical")):
                return res[0, "transcript_name"]
            return df.sort("annotation", descending=False)[0, "transcript_name"]

        tss = []
        for gene in corrected:
            try:
                tss.append(parse(get_transcripts(ds, gene, mode="appris")))
            except pl.ComputeError:
                tss.append(parse(get_transcripts(ds, gene, mode="ensembl")))

        path.with_suffix(".tss.txt").write_text("\n".join(sorted(tss)))
    else:
        corrected = tss = path.read_text().splitlines()

    if existing:
        if single:
            raise ValueError("Cannot use --single with existing codebook")
        assert existing_cb
        genes = set(ts.split("-")[0] for ts in existing_cb)
        old = set(corrected)
        corrected = old - genes
        logger.info(f"{len(corrected)} genes remaining after removing {old & set(genes)}")

    if existing:
        assert existing_cb
        used_bits = set(chain.from_iterable(existing_cb.values()))
        offset = max(order.index(x) if x in order else -1 for x in used_bits) + 1
    else:
        offset = 0

    cb = (
        gen_codebook(tss, offset=offset) if not single else {x: [order[i]] for i, x in enumerate(tss, offset)}
    )

    Path(f"{path.stem.split('.')[0]}.json").write_text(json.dumps(cb, default=int))
    return cb


# %%
# out = "starwork2/10xhuman.txt"
# Path(out).write_text("\n".join(res := sorted(list(set(tenx["Genes"]) | set(cs)))))


# %%

# %%
# out = set(Path("starwork2/zach.tss.txt").read_text().splitlines()) | set(
#     Path("starwork2/tricycle.tss.txt").read_text().splitlines()
# ) - set(Path("starwork2/genestar.tss.txt").read_text().splitlines())
# # %%
# Path("starwork2/zach.tss.txt").write_text("\n".join(sorted(out)))
# %%
# fpkm = pl.read_csv("../oligocheck/data/fpkm/combi99percentile.csv")
# fpkm = dict(zip(fpkm["gene"], fpkm["value"]))
# # %%
# Path("starwork2/tricycleplus.txt").read_text().splitlines()

# %%
# # %%
# diff = """Vim
# Rpsa
# Grb10
# Ackr3
# Ccnd2
# Chd7
# Boc
# Cdon
# Rpl12
# H2afz
# Rps27l
# Gnb2l1
# Hes1
# Plagl1
# Psat1
# Hsph1
# Rbm20
# Polr2a
# Nes
# Ldha
# Clk2
# 2410131K14Rik
# Rapgef6
# Bcl11b
# Zfp618
# Nfia
# Tanc1
# Cenpa
# Acly
# Cadps
# Clec11a
# Rtel1
# Kcna1
# Hmga2
# Qk
# Etl4
# Sf3b3
# Pitrm1
# Eml5
# Prkd3
# Naa16
# Atg12
# Xist
# Akt2
# Ddx19b
# Med1
# Arx
# Golim4
# Trps1
# Unc5d
# Ube2ql1
# Aff2
# Sparcl1
# Kif26b
# Enc1
# Crtc3
# Osbpl6
# Syt4
# Celf2
# Nlgn3
# Atp8a1
# Sox11
# Ifitm3
# Dnm3
# Trim67
# Nrxn1
# Zbtb18
# Ptn
# Cd24a
# Mpped1
# Dkk3
# Dpysl3
# A530017D24Rik
# Mllt11
# Bcl11a
# Ptprk
# Sema3c
# Cadm2
# Camk2b
# Neurod6
# Satb2
# Gm17750
# Dcx
# Ppfia2
# Mapt
# Abca1
# Clstn2
# Tuba1a
# Ttc28
# Arpp21
# Gpm6a
# Smarca2
# Rtn1
# Neurod2
# Zbtb20
# Gria2
# Tubb3
# Prkar1b
# Mef2c""".splitlines()

# # %%
# birth = """Hmga2
# Tbr1
# Tbc1d16
# Top2a
# Cabp1
# Hsph1
# Fn1
# Bcl11b
# Rpsa
# Nt5dc3
# Nusap1
# Qars
# H2afz
# Ina
# Gadd45g
# Zfp618
# Prpf4
# Ube2c
# Gria1
# Rpa1
# Hes1
# 6430573F11Rik
# Siva1
# RP23-379C24.2
# Gfm2
# Lrrc9
# Chaf1b
# Stat4
# Ptpro
# Map4k2
# Nat8l
# Eml5
# Syt4
# Prkg2
# Gm13157
# Nfia
# Rangap1
# Leprel1
# Ap1g2
# Shisa6
# Dgkg
# Nfatc2
# Filip1
# Smim12
# Fndc3c1
# Map1b
# Gabpb2
# Sox5
# Tmem108
# Nrxn1
# Cd24a
# Acadvl
# Cnksr2
# Vwa5b2
# Nedd4
# Ncam1
# Glt28d2
# Ptprz1
# Meis2
# Osbpl6
# Stk32a
# RP23-14P23.9
# Jakmip3
# Cttnbp2
# Sparcl1
# Crtc3
# Ddah1
# Glra2
# Smim14
# Dpysl3
# mt-Nd2
# Apoe
# Trim9
# Clu
# Nr2f1
# Smarca2
# Lgals1
# Clstn1
# Bcan
# Trim2
# Ttc28
# Atp1b1
# Mir99ahg
# Tnc
# Chl1
# Thoc2
# Ndrg2
# Dbi
# Pantr1
# Ptprk
# Gm17750
# Unc5d
# Ptn
# Sema3c
# Slc1a3
# Mfge8
# Zbtb20
# Fabp7
# Yam1""".splitlines()
# # %%
# tricycleplus = set(
#     {x: fpkm.get(x.split("-")[0], None) for x in diff if (fpkm.get(x, None) or 0) < 40}
#     | {x: fpkm.get(x.split("-")[0], None) for x in birth if (fpkm.get(x, None) or 0) < 40}
#     | {x: fpkm.get(x.split("-")[0], None) for x in Path("starwork2/tricycle.txt").read_text().splitlines()}
# ) - set(Path("starwork2/genestar.txt").read_text().splitlines())

# Path("starwork2/tricycleplus.txt").write_text("\n".join(sorted(list(tricycleplus))))
# # %%
if __name__ == "__main__":
    main()

# %%
# tricycleplus = set(
#     # {x: fpkm.get(x.split("-")[0], None) for x in diff if (fpkm.get(x, None) or 0) < 40}
#     # | {x: fpkm.get(x.split("-")[0], None) for x in birth if (fpkm.get(x, None) or 0) < 40}
#     {x: fpkm.get(x.split("-")[0], None) for x in Path("starwork2/tricycleplus.txt").read_text().splitlines()}
# ) - set(Path("starwork2/genestar.txt").read_text().splitlines())

# Path("starwork2/tricycleplus.txt").write_text("\n".join(sorted(list(tricycleplus))))

# # %%
# tricycleplus = set(Path("starwork2/laiorganoid.converted.txt").read_text().splitlines()) - set(
#     Path("starwork2/genestar.converted.txt").read_text().splitlines()
# )

# # %%
# Path("starwork2/laiorganoid.txt").write_text("\n".join(sorted(list(tricycleplus))))
# %%
