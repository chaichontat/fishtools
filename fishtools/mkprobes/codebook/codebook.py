# %%
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


class CodebookPicker:
    def __init__(
        self,
        mhd4: npt.NDArray[np.bool_] | str | Path,
        subset: tuple[int, int] | None = None,
        counts_existing: npt.NDArray[Any] | None = None,
        code_existing: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        if isinstance(mhd4, str | Path):
            mhd4 = np.loadtxt(mhd4, delimiter=",", dtype=bool)
        self.mhd4 = mhd4
        self.subset = subset

        if (counts_existing is not None) ^ (code_existing is not None):
            raise ValueError("Either both counts and code must be provided or neither")

        self.code_existing = (
            np.zeros((0, self.mhd4.shape[1]), dtype=bool) if code_existing is None else code_existing
        )
        assert self.code_existing.shape[1] == self.mhd4.shape[1]
        self.used_code = {tuple(row) for row in self.code_existing}
        if code_existing is not None and counts_existing is not None:
            self.counts_existing = np.percentile(counts_existing @ code_existing, 99.99, axis=0)
        else:
            self.counts_existing = None

        self.r_ = (
            np.r_[0 : self.subset[0], self.subset[1] : self.mhd4.shape[1]]
            if self.subset is not None
            else None
        )

    def gen_codebook(self, seed: int):
        rand = np.random.RandomState(seed)
        mhd4 = self.mhd4[self.mhd4[:, self.r_].sum(axis=1) == 0] if self.subset is not None else self.mhd4
        rmhd4 = np.array(list({tuple(row) for row in mhd4} - self.used_code))
        rand.shuffle(rmhd4)
        return rmhd4

    # def _find_optimalish(self, seed: int, fpkm: Sized):
    #     rmhd4 = self.gen_codebook(seed)
    #     res = rmhd4[: len(fpkm)] * np.array(fpkm)
    #     tocalc = res.sum(axis=0)
    #     normed = tocalc / tocalc.sum()
    #     return -np.sum(normed * np.log2(normed)), tocalc

    # def find_optimalish(self, fpkm: Sized, iterations: int = 5000):
    #     res = [self._find_optimalish(i, fpkm)[0] for i in range(iterations)]
    #     best = np.argmax(res)
    #     return best, self._find_optimalish(int(best), fpkm)[1]


class CodebookPickerSingleCell(CodebookPicker):
    def find_optimalish(
        self,
        counts: npt.NDArray[Any],
        *,
        iterations: int = 1000,
    ):
        def _find(seed: int):
            if seed % 1000 == 0:
                print(seed)
            rmhd4 = self.gen_codebook(seed)

            # combicode[: self.existing.shape[0], : self.existing.shape[1]] = self.existing
            # combicode[self.existing.shape[0] :] = rmhd4
            tocalc = np.percentile((counts @ rmhd4[: counts.shape[1]]), 99.99, axis=0)
            if self.counts_existing is not None:
                tocalc = np.vstack([self.counts_existing, tocalc])
            normed = tocalc / tocalc.sum()
            # if not np.all(normed > 0):
            #     print(np.where(normed == 0)[0], "is 0")
            normed = normed[normed > 0]
            return -np.sum(normed * np.log2(normed)), tocalc

        res = [_find(i)[0] for i in range(iterations)]
        best = np.argmax(res)
        return best, _find(int(best))[1]


def what():
    name = 'sm'
readouts = pl.read_csv("data/readout_ref_filtered.csv")
# genes = Path("zach_26.txt").read_text().splitlines()
# genes = Path(f"{name}_25.txt").read_text().splitlines()
smgenes = ['Cdc42','Neurog2','Ccnd2']
genes, _, _ = gtf_all.check_gene_names(smgenes)
acceptable_tss = {g: set(pl.read_csv(f"output/{g}_acceptable_tss.csv")["transcript"]) for g in genes}
n, short_threshold = 67, 65
# %%
dfx, overlapped = {}, {}
for gene in genes:
    dfx[gene] = GeneFrame.read_parquet(f"output/{gene}_final.parquet")
dfs = GeneFrame.concat(dfx.values())
short = dfs.count("gene").filter(pl.col("count") < short_threshold)


# %%
fixed_n = {}
short_fixed = {}


def run_overlap(genes: Iterable[str], overlap: int):
    def runpls(gene: str):
        subprocess.run(
            ["python", "scripts/new_postprocess.py", gene, "-O", str(overlap)],
            check=True,
            capture_output=True,
        )

    with ThreadPoolExecutor(32) as executor:
        for x in as_completed(
            [
                executor.submit(runpls, gene)
                for gene in genes
                if not Path(f"output/{gene}_final_overlap_{overlap}.parquet").exists()
            ]
        ):
            print("ok")
            x.result()


    needs_fixing = set(short["gene"])

    for ol in [5, 10, 15, 20]:
        print(ol, needs_fixing)
        run_overlap(needs_fixing, ol)
        for gene in needs_fixing.copy():
            df = GeneFrame.read_parquet(f"output/{gene}_final_overlap_{ol}.parquet")
            if len(df) >= short_threshold or ol == 20:
                needs_fixing.remove(gene)
                fixed_n[gene] = ol
                short_fixed[gene] = df
    # else:
    #     raise ValueError(f"Gene {gene} cannot be fixed")

    short_fixed = GeneFrame.concat(short_fixed.values())
    # %%
    cutted = GeneFrame.concat([dfs.filter(~pl.col("gene").is_in(short["gene"])), short_fixed[dfs.columns]])
    cutted = GeneFrame(
        cutted.sort(["gene", "priority"]).groupby("gene").agg(pl.all().head(n)).explode(pl.all().exclude("gene"))
    )