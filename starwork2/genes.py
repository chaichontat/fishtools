# %%
import json
from pathlib import Path
from subprocess import run

import polars as pl

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
Wnt4""".splitlines()
tenx = pl.read_csv("static/10x.csv")
# %%
out = "starwork2/genestar.txt"
Path(out).write_text("\n".join(res := sorted(list(set(tenx["Genes"]) | set(cs) | set(merfish[:120])))))

run(f"mkprobes chkgenes data/mouse {out}", shell=True, check=True)

# %%
ds = Dataset("data/mouse")
corrected = Path(out).with_suffix(".converted.txt").read_text().splitlines()


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
# %%
Path(out).with_suffix(".tss.txt").write_text("\n".join(tss))


# %%


def gen_cb(mode: str):
    if mode == "tricycle":
        offset = 5
        n = 12
    elif mode == "genestar":
        offset = 1
        n = 15
    else:
        raise ValueError

    mapping = {}
    for i in range(n):
        d, m = divmod(i, 4)
        mapping[i + 1] = 8 * d + offset + m

    genes = Path(f"starwork2/{mode}.tss.txt").read_text().splitlines()
    cb = CodebookPicker(f"static/{n}bit_on3_dist2.csv", genes=genes)
    cb.gen_codebook(1)
    c = cb.export_codebook(1, offset=1)
    Path(f"{mode}.json").write_text(
        json.dumps({k: list(map(mapping.get, v)) for k, v in c.items()}, default=int)
    )


gen_cb("genestar")
