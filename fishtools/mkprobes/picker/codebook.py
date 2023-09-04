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
