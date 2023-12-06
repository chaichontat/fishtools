# %%
import json
from pathlib import Path
from subprocess import run
from typing import Literal

import click
import polars as pl
from loguru import logger

from fishtools.mkprobes.codebook.codebook import CodebookPicker
from fishtools.mkprobes.ext.external_data import Dataset
from fishtools.mkprobes.genes.chkgenes import get_transcripts

merfish = """1700022I11Rik
1810046K07Rik
5031425F14Rik
5730522E02Rik
Acta2
Adam2
Adamts2
Adamts4
Adra1b
Alk
Ankfn1
Ano4
Aqp4
Asic4
B4galnt2
B4galnt3
Barx2
Bcl11b
Bdnf
Bgn
Blnk
Bmpr1b
Brinp3
C1ql3
C1qtnf7
Cacng5
Calb2
Camk2d
Car3
Cbln2
Cbln4
Ccbe1
Ccdc162
Ccdc3
Ccdc80
Ccnb1
Cd14
Cd24a
Cdca7
Cdcp1
Cdh12
Cdh13
Cdh20
Cdh9
Ceacam9
Cemip
Chat
Chn2
Chodl
Chrm2
Chrna2
Cldn5
Clrn1
Cntnap5b
Cobll1
Col14a1
Col15a1
Col23a1
Col24a1
Col25a1
Corin
Cplx3
Crhr2
Crispld2
Cspg4
Ctss
Cux2
Cxcl14
Daam2
Dmkn
Dnase1l3
Dscaml1
Egfem1
Egfr
Egln3
Egr2
Elfn1
Enpp6
Epha7
Fam19a2
Fam84b
Fbxl7
Fezf2
Flrt3
Flt1
Fndc7
Fosb
Foxp2
Frem2
Fst
Gfap
Glra1
Gpc6
Grik1
Grin3a
Grm1
Grm8
Hpse
Hs3st5
Igf2
Igfbp4
Igfbp5
Ikzf2
Il1rapl2
Il4ra
Inpp4b
Iqgap2
Itgb8
Kcng1
Kcnj8
L3mbtl4
Lama3
Lhx6
Lmo1
Lsp1
Ltf
Luzp2
Lypd1
Lyzl4
Marchf1
Marcksl1
Meis2
Moxd1
Mrc1
Mrgprx2
Muc20
Myh14
Ndst4
Nhs
Nkain3
Nnmt
Nos1
Npas1
Npnt
Npsr1
Npy2r
Nr2f2
Nr4a1
Nr4a2
Ntng2
Nxph1
Nxph2
Nxph4
Olah
Olfm3
Opalin
Oprk1
Osr1
Otof
Parm1
Pcdh8
Pde11a
Pdgfc
Pdgfra
Pdlim5
Phactr2
Plch1
Plcxd3
Pld5
Plekhg3
Pou3f1
Pou3f3
Pou6f2
Prdm8
Prok2
Prokr2
Prox1
Prr16
Prss12
Prss23
Ptger3
Ptprk
Ptprm
Ptprt
Ptpru
Pxdc1
Ramp1
Reln
Rerg
Rfx3
Rgs5
Rgs6
Rnf152
Ror1
Rorb
Rspo1
Rxfp1
Rxfp2
Satb2
Scgn
Sema3e
Sema5a
Serpinf1
Sertm1
Sgcd
Shisa9
Slc17a6
Slc17a8
Slc25a13
Slc30a3
Slc32a1
Slc44a5
Slco5a1
Sncg
Sox10
Sox6
Sp8
Spon1
St6galnac5
Sulf1
Sulf2
Syndig1
Syt10
Syt6
Tbc1d4
Tcap
Teddm3
Tenm3
Th
Thbs2
Thsd7a
Timp3
Tmem163
Tmtc2
Tnfaip6
Tox
Trp53i11
Trpc4
Trpc6
Tshz2
Tunar
Ubash3b
Unc13c
Unc5b
Unc5d
Ust
Vipr2
Vtn
Vwc2
Wipf3
Wnt7b
Zfp804b
Vip
Sst
Calb1
Rab3b
Gad2
Slc17a7
Tac2
Penk
Lamp5
Cd52
Rprml
Mup5
Cnr1
Gad1
Pvalb
Igfbp6""".splitlines()

cs = """Ascl1
Sox9
Sox2
Vim
Chgb
Rprm
Akap7
Drd1
Fgf10
Nes
Bcl11b
Tbr2
Ctip2
Mef2c
Pde1a
Prox1
Nwd2
Cbln1
Slc6a11
Mfge8
Htra1
Gpr17
Opalin
Btg2
Pax6
Foxg1
Ccl4
Ccl3
Sost
Tac1
Cnksr3
Dmkn
Foxp2
Scgn
Gadd45g
Sox5
Ldb2
Satb2
Neurod1
Neurod2
Neurod6
Nrp1
Hes1
Hes5
Neurog2
Tle4
Satb2
Dcx
Gad1
Cd24a
Pvalb
Gabrr2
Hoxb6
Bhlhe22
Nfix
Lamp5
Unc5d
Slc18a2
Slc6a3
Oxt
Hcrt
Nefm
Sema3e
Tcerg1l
Zeb1
Pou2f2
Pou3f1
Rorb
Fam19a2
Cdh13
PlxnD1
Igfbpl1
Igsf21
Gnb4
Pcp4
Slc1a3
Foxj1
Dmrta2
Myt1l
Nr2f1
Lhx2
Emx1
Emx2
Tcf3
Jun
Sox4
Sox11
Nfix
Mef2c
Thra
Hmgb2
Eif1b
2900055J20Rik
Atp1b1
Ddah1
Cux1
Fezf2
Etv1
Ptn
Meg3
Pantr1
Nrgn
Snca
Tshz2
Tshz3
Gng8
Chgb
Tcl4
Birc5
Calm1
Npy
Nnat
Alyref
Slc1a3
Rgs16
Nts
Mef2c
Unc5d
Ldb2
Sox5
Nes
Fn1
Sall4
Nr2f1
Ephb1
Wnt4
Dll1
Dll3
Notch1
Notch2
Tgfbr1
Irf6
Smad2
Smad3""".splitlines()
# tenx = pl.read_csv("static/10x.csv")


@click.command()
@click.argument("data", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.argument("path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option("--existing", "-e", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--species", type=click.Choice(["mouse", "human"]), default="mouse")
@logger.catch
def main(data: Path, path: Path, existing: Path, species: Literal["mouse", "human"]):
    if "converted" in path.name:
        raise ValueError("Path should be to the original file")
    # if not path.with_suffix(".converted.txt").exists():
    path.with_suffix(".converted.txt").unlink(missing_ok=True)
    run(f"mkprobes chkgenes {data} {path} --species {species}", shell=True, check=True)
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

    if existing:
        # set(Path("starwork2/genestar.converted.txt").read_text().splitlines())
        genes = set(ts.split("-")[0] for ts in json.loads(existing.read_text()))
        old = set(corrected)
        corrected = old - genes
        logger.info(f"{len(corrected)} genes remaining after removing {old & set(genes)}")

    tss = []
    for gene in corrected:
        try:
            tss.append(parse(get_transcripts(ds, gene, mode="appris")))
        except pl.ComputeError:
            tss.append(parse(get_transcripts(ds, gene, mode="ensembl")))

    path.with_suffix(".tss.txt").write_text("\n".join(sorted(tss)))


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
