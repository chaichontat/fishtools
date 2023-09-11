# %%
from pathlib import Path
from typing import Any, Sized

import numpy as np
import numpy.typing as npt
import polars as pl
from loguru import logger


class CodebookPicker:
    def __init__(
        self,
        mhd4: npt.NDArray[np.bool_] | str | Path,
        genes: list[str],
        subset: tuple[int, int] | None = None,
        counts_existing: npt.NDArray[Any] | None = None,
        code_existing: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        if isinstance(mhd4, str | Path):
            mhd4 = np.loadtxt(mhd4, delimiter=",", dtype=bool)

        self.mhd4 = mhd4
        if not np.all(np.logical_or(self.mhd4 == 0, self.mhd4 == 1)):
            raise ValueError("MHD4 must only contain 0 and 1")

        self.genes = genes
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

    def _find_optimalish(self, seed: int, fpkm: Sized):
        rmhd4 = self.gen_codebook(seed)
        res = rmhd4[: len(fpkm)] * np.array(fpkm)
        tocalc = res.sum(axis=0)
        normed = tocalc / tocalc.sum() + 1e-10

        return -np.sum(normed * np.log2(normed)), tocalc

    def export_codebook(self, seed: int) -> pl.DataFrame:
        rmhd4 = self.gen_codebook(seed)
        n_blanks = self.mhd4.shape[0] - len(self.genes)
        return pl.concat(
            [
                pl.DataFrame(dict(genes=self.genes + [f"Blank-{i+1}" for i in range(n_blanks)])),
                pl.DataFrame(rmhd4.astype(np.uint8)),
            ],
            how="horizontal",
        )


class CodebookPickerSingleCell(CodebookPicker):
    def find_optimalish(
        self,
        counts: npt.NDArray[Any],
        *,
        iterations: int = 200,
        percentile: float = 99.9,
    ):
        def _find(seed: int):
            rmhd4 = self.gen_codebook(seed)

            # combicode[: self.existing.shape[0], : self.existing.shape[1]] = self.existing
            # combicode[self.existing.shape[0] :] = rmhd4
            # numpy bug??
            tocalc = np.asarray(
                np.percentile(
                    (counts @ rmhd4[: counts.shape[1]]), np.full(self.mhd4.shape[1], percentile), axis=0
                )[0]
            ).squeeze()
            if self.counts_existing is not None:
                tocalc = np.vstack([self.counts_existing, tocalc])
            normed = (tocalc + 1e-10) / tocalc.sum()
            return -np.sum(normed * np.log2(normed)), tocalc

        if counts.shape[1] != len(self.genes):
            raise ValueError("Mismatch array size between gene name list and counts matrix")

        if counts.shape[1] > self.mhd4.shape[0]:
            raise ValueError("Number of genes is larger than the number of possible codes")

        if counts.shape[1] > 0.95 * self.mhd4.shape[0]:
            logger.warning(
                f"Number of genes ({counts.shape[1]}) is close to the number of possible codes ({self.mhd4.shape[1]}). "
                "This may result in a suboptimal codebook. "
                "Consider using a larger codebook or a smaller number of genes."
            )

        res = [_find(i)[0] for i in range(iterations)]
        best = np.argmax(res)
        logger.info(
            f"Best codebook found at seed {best} with entropy {res[best]:.3f} (worst entropy is {np.min(res):.3f})."
        )

        return best, _find(int(best))[1]
