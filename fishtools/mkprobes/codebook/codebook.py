# %%
import json
from hashlib import md5
from pathlib import Path
from typing import Any, Collection, Literal, Sized, overload

import numpy as np
import numpy.typing as npt
import polars as pl
from loguru import logger
from pydantic import BaseModel, TypeAdapter


def hash_codebook(cb: dict[str, Collection[int]]) -> str:
    return md5(
        json.dumps(cb, sort_keys=True, ensure_ascii=True).encode(),
        usedforsecurity=False,
    ).hexdigest()[-6:]


class CodebookPicker:
    def __init__(
        self,
        mhd4: npt.NDArray[np.bool_] | str | Path,
        genes: list[str],
        subset: tuple[int, int] | None = None,
        existing: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        if isinstance(mhd4, str | Path):
            mhd4 = np.loadtxt(mhd4, delimiter=",", dtype=bool)

        self.mhd4 = mhd4
        if not np.all(np.logical_or(self.mhd4 == 0, self.mhd4 == 1)):
            raise ValueError("MHD4 must only contain 0 and 1")

        self.genes = genes
        self.subset = subset
        self.existing = existing

        if self.existing is not None:
            self.existing = np.hstack(
                [
                    self.existing,
                    np.zeros(
                        (self.existing.shape[0], self.mhd4.shape[1] - self.existing.shape[1]), dtype=bool
                    ),
                ]
            )
            assert self.existing.shape[1] == self.mhd4.shape[1]
            assert np.all(self.existing.sum(axis=1) == 4)
            self.mhd4 = np.array(
                list({tuple(row) for row in self.mhd4} - {tuple(row) for row in self.existing})
            )

        # if code_existing is not None and counts_existing is not None:
        #     self.counts_existing = np.percentile(counts_existing @ code_existing, 99.99, axis=0)
        # else:
        #     self.counts_existing = None

        # self.r_ = (
        #     np.r_[0 : self.subset[0], self.subset[1] : self.mhd4.shape[1]]
        #     if self.subset is not None
        #     else None
        # )

    def gen_codebook(self, seed: int):
        rand = np.random.RandomState(seed)
        # mhd4 = self.mhd4[self.mhd4[:, self.r_].sum(axis=1) == 0] if self.subset is not None else self.mhd4
        rmhd4 = self.mhd4.copy()
        rand.shuffle(rmhd4)
        return rmhd4

    def _calc_entropy(self, seed: int, fpkm: Sized):
        rmhd4 = self.gen_codebook(seed)
        res = rmhd4[: len(fpkm)] * np.array(fpkm).reshape(-1, 1)
        tocalc = res.sum(axis=0)
        normed = tocalc / tocalc.sum() + 1e-10

        return -np.sum(normed * np.log2(normed)), tocalc

    def find_optimalish(self, fpkm: npt.NDArray[Any], iterations: int = 200):
        if fpkm.size != len(self.genes):
            raise ValueError("Mismatch array size between gene name list and counts matrix")

        if fpkm.size > self.mhd4.shape[0]:
            raise ValueError("Number of genes is larger than the number of possible codes")

        if fpkm.size > 0.95 * self.mhd4.shape[0]:
            logger.warning(
                f"Number of genes ({fpkm.size}) is close to the number of possible codes ({self.mhd4.shape[1]}). "
                "This may result in a suboptimal codebook. "
                "Consider using a larger codebook or a smaller number of genes."
            )

        res = [self._calc_entropy(i, fpkm)[0] for i in range(iterations)]
        best = np.argmax(res)
        logger.info(
            f"Best codebook found at seed {best} with entropy {res[best]:.3f} (worst entropy is {np.min(res):.3f})."
        )

        return int(best), self._calc_entropy(int(best), fpkm)[1]

    @overload
    def export_codebook(
        self, seed: int, type: Literal["json"] = ..., offset: int = ...
    ) -> dict[str, list[int]]:
        ...

    @overload
    def export_codebook(self, seed: int, type: Literal["csv"], offset: int = ...) -> pl.DataFrame:
        ...

    def export_codebook(
        self, seed: int, type: Literal["csv", "json"] = "json", offset: int = 1
    ) -> pl.DataFrame | dict[str, list[int]]:
        rmhd4 = self.gen_codebook(seed)
        n_blanks = self.mhd4.shape[0] - len(self.genes)
        match type:
            case "csv":
                return pl.concat(
                    [
                        pl.DataFrame(dict(genes=self.genes + [f"Blank-{i + 1}" for i in range(n_blanks)])),
                        pl.DataFrame(rmhd4.astype(np.uint8)),
                    ],
                    how="horizontal",
                )
            case "json":
                return {
                    gene: sorted(np.flatnonzero(code) + offset)
                    for gene, code in zip(
                        self.genes + [f"Blank-{i + 1}" for i in range(n_blanks)], rmhd4.astype(int)
                    )
                }
            case _:  # type: ignore
                raise ValueError(f"Unknown type {type}")


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


class ProbeSet(BaseModel):
    name: str
    species: str
    codebook: str
    bcidx: int
    existing: str | None = None
    single: bool = False
    all_bit: int = 29
    n_probes: Literal["high", "low"] | None = None

    def load_codebook(self, path: Path | str):
        path = Path(path)
        return {
            k: v
            for k, v in json.loads((path / self.codebook).read_text()).items()
            if not k.startswith("Blank")
        }

    def codebook_dfs(self, path: Path | str):
        codebook = self.load_codebook(path)
        tss = list(codebook)
        dfs = pl.concat(
            [
                pl.read_parquet(Path(path) / f"output/{ts}_final_BamHIKpnI_{hash_codebook(codebook)}.parquet")
                # .sample(shuffle=True, seed=4, fraction=1)
                .sort(["priority", "hp"])
                for ts in tss
            ]
        )
        return dfs

    @classmethod
    def from_list_json(cls, path: str | Path):
        return TypeAdapter(list[cls]).validate_json(Path(path).read_text())

    @classmethod
    def from_manifest(cls, path: str | Path):
        return cls.from_list_json(path)
