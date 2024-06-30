from __future__ import annotations

from functools import reduce, wraps
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, ParamSpec, TypeVar

import polars as pl
from loguru import logger
from typing_extensions import Self

from fishtools.utils.utils import copy_signature, copy_signature_method

from ._alignment import run_bowtie
from .seqcalc import tm_pairwise

T = TypeVar("T")
P, R = ParamSpec("P"), TypeVar("R")


def to_samframe(f: Callable[P, pl.DataFrame]) -> Callable[P, SAMFrame]:
    @wraps(f)
    def wrap(*args: P.args, **kwargs: P.kwargs) -> SAMFrame:
        return SAMFrame(f(*args, **kwargs))

    return wrap


class LazySAMFrame(pl.LazyFrame):
    def __init__(self, ldf: pl.LazyFrame):
        self._ldf = ldf._ldf

    @to_samframe
    @copy_signature(pl.LazyFrame.collect)
    def collect(self, *args: Any, **kwargs: Any):
        return super().collect(*args, **kwargs)


class SAMFrame(pl.DataFrame):
    # NECESSARY_COLUMNS = {"seq", "transcript", "pos_start", "pos_end"}
    def __init__(self, df: pl.DataFrame):
        self._df = df._df

    def lazy(self) -> LazySAMFrame:
        return LazySAMFrame(super().lazy())

    @to_samframe
    def count(self, col: str = "gene", descending: bool = False):
        return self.groupby(col).agg(pl.count()).sort("count", descending=descending)

    @to_samframe
    @copy_signature(pl.DataFrame.join)
    def join(self, *args: Any, **kwargs: Any):
        return super().join(*args, **kwargs)

    @to_samframe
    @copy_signature(pl.DataFrame.sort)
    def sort(self, *args: Any, **kwargs: Any):
        return super().sort(*args, **kwargs)

    def sort_(self, **kwargs: bool):
        return self.sort(by=list(kwargs.keys()), descending=list(kwargs.values()))

    @to_samframe
    @copy_signature(pl.DataFrame.filter)
    def filter(self, *args: Any, **kwargs: Any):
        return super().filter(*args, **kwargs)

    def gene(self, gene: str):
        return self.filter(pl.col("gene") == gene)

    def filter_eq(self, **kwargs: str | float):
        return self.filter(reduce(lambda x, y: x & y, [pl.col(k) == v for k, v in kwargs.items()]))

    def filter_isin(self, **kwargs: Collection[str] | pl.Series):
        return self.filter(reduce(lambda x, y: x & y, [pl.col(k).is_in(v) for k, v in kwargs.items()]))

    def left_join(
        self,
        other: pl.DataFrame,
        on: str | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
    ):
        return self.join(other, on=on, left_on=left_on, right_on=right_on, how="left")

    @classmethod
    def concat(cls, dfs: Iterable[pl.DataFrame]) -> SAMFrame:
        return cls(pl.concat(dfs))

    @classmethod
    def read_parquet(cls, path: str | Path):
        return cls(pl.read_parquet(path))

    @staticmethod
    def _count_match(df: pl.DataFrame) -> SAMFrame:
        return SAMFrame(
            df.join(
                df[["id", "mismatched_reference"]]
                .with_columns(mismatched_reference=pl.col("mismatched_reference").str.extract_all(r"(\d+)"))
                .explode("mismatched_reference")
                .with_columns(pl.col("mismatched_reference").cast(pl.UInt8))
                .groupby("id")
                .agg(
                    match=pl.col("mismatched_reference").sum(),
                    match_consec=pl.col("mismatched_reference").max(),
                ),
                on="id",
                how="left",
            )
        )

    @classmethod
    def from_sam(cls, sam: str, split_name: bool = True, count_match: bool = True) -> SAMFrame:
        # s = (
        #     pl.DataFrame(dict(strs=[sam]))
        #     .lazy()
        #     .with_columns(pl.col("strs").str.split("\n"))
        #     .explode("strs")
        #     .with_columns(pl.col("strs").str.strip().str.split("\t").list.slice(0, 10))
        #     .with_row_count("id")
        #     .explode("strs")
        #     .with_columns(col_nm="string_" + pl.arange(0, pl.count()).cast(pl.Utf8).str.zfill(2).over("id"))
        #     .sort(["col_nm", "id"])
        #     .collect()
        # )

        # faster than pivot
        # key_optional = {
        #     "AS": "aln_score",
        #     "XS": "aln_score_best",
        #     "XN": "n_ambiguous",
        #     "XM": "n_mismatches",
        #     "XO": "n_opens",
        #     "XG": "n_extensions",
        #     "NM": "edit_distance",
        #     "MD": "mismatched_reference",
        #     "YT": "pair_state",
        # }

        # fmt: off
        logger.info(f"Parsing SAM output. Length of SAM: {len(sam.splitlines())}")
        df = (
            pl.read_csv(StringIO(sam), separator="\n", has_header=False)
            .lazy()
            .with_row_count("id")
            .with_columns(temp=pl.col("column_1").str.split_exact("\t", 9))
            .unnest("temp")
            .rename(
                {
                    f"field_{i}": x
                    for i, x in enumerate(["name", "flag", "transcript", "pos", "mapq", "cigar", "rnext", "pnext", "tlen", "seq"])
                }
            )

            .with_columns(
                flag=pl.col("flag").cast(pl.UInt16),
                pos=pl.col("pos").cast(pl.UInt32),
                mapq=pl.col("mapq").cast(pl.UInt8),
                aln_score=pl.col("column_1").str.extract(r"AS:i:(\d+)").cast(pl.UInt16),
                aln_score_best=pl.col("column_1").str.extract(r"XS:i:(\d+)").cast(pl.UInt16),
                n_ambiguous=pl.col("column_1").str.extract(r"XN:i:(\d+)").cast(pl.UInt16),
                n_mismatches=pl.col("column_1").str.extract(r"XM:i:(\d+)").cast(pl.UInt16),
                n_opens=pl.col("column_1").str.extract(r"XO:i:(\d+)").cast(pl.UInt16),
                n_extensions=pl.col("column_1").str.extract(r"XG:i:(\d+)").cast(pl.UInt16),
                edit_distance=pl.col("column_1").str.extract(r"NM:i:(\d+)").cast(pl.UInt16),
                mismatched_reference=pl.col("column_1").str.extract(r"MD:Z:(\S+)"),
            )
            .drop(["column_1", "mapq", "rnext", "pnext", "tlen"])
            .with_columns(
                [
                    pl.when(pl.col("transcript").str.contains(r"(.*)\.\d+"))
                    .then(pl.col("transcript").str.extract(r"(.*)\.\d+"))
                    .otherwise(pl.col("transcript"))
                    .alias("transcript")
                ]
                + [
                    pl.col("name").str.extract(r"(.+)_(.+):(\d+)-(\d+)", 1).alias("gene"),
                    pl.col("name").str.extract(r"(.+)_(.+):(\d+)-(\d+)", 2).alias("transcript_ori"),
                    pl.col("name").str.extract(r"(.+)_(.+):(\d+)-(\d+)", 3).cast(pl.UInt32).alias("pos_start"),
                    pl.col("name").str.extract(r"(.+)_(.+):(\d+)-(\d+)", 4).cast(pl.UInt32).alias("pos_end"),
                ]
                if split_name
                else []
            )
            .with_columns(
                [(pl.col("pos_end") - pl.col("pos_start") + 1).alias("length")]
                if split_name
                else []
            )
            .collect()
        )
        return cls._count_match(df) if count_match else cls(df)
        # fmt: on

    def filter_by_match(
        self,
        acceptable_tss: Collection[str],
        match: float = 0.8,
        match_consec: float = 0.8,
        min_length: int = 100,
        min_consec_length: int = 100,
    ) -> SAMFrame:
        """Cannot do a groupby because this is a percent comparison.
        Each transcript must be compared to its length.

        Percent match or minimum length, which ever is lower."""

        # Filter by match and match_consec.
        nogo = self.filter(
            (
                (pl.col("match") > pl.col("length") * match)
                | (pl.col("match") > min_length)
                | (pl.col("match_consec") > pl.col("length") * match_consec)
                | (pl.col("match_consec") > min_consec_length)
            )
            & ~pl.col("transcript").is_in(acceptable_tss)
        )

        return SAMFrame(self.filter(~pl.col("name").is_in(nogo["name"])))

    @to_samframe
    def agg_tm_offtarget(self, acceptable_tss: Collection[str], formamide: int = 10):
        # Filter by maximum offtarget Tms.
        # Only calculate Tm for sequences with >15 consecutive matches
        # to save time.

        tm_offtarget = self.filter(
            ~pl.col("transcript").is_in(acceptable_tss) & pl.col("match_consec").gt(18)
        ).with_columns(
            tm_offtarget=pl.struct(["seq", "cigar", "mismatched_reference"]).apply(
                lambda x: tm_pairwise(x["seq"], x["cigar"], x["mismatched_reference"], formamide=formamide)  # type: ignore
            )
        )

        unique = tm_offtarget.groupby("name").agg(
            pl.max("tm_offtarget").alias("max_tm_offtarget").cast(pl.Float32),
            pl.max("match_consec").alias("match_consec_all"),
        )
        # Aggregate
        out = (
            self.filter(pl.col("transcript").is_in(acceptable_tss))
            .unique("name")
            .join(unique, on="name", how="left")
            .with_columns(
                max_tm_offtarget=pl.col("max_tm_offtarget").fill_null(0.0),
                match_consec_all=pl.col("match_consec_all").fill_null(0),
            )
        )

        if not len(out):
            logger.critical(
                "No transcripts found during aggregate. Most likely due to wrong acceptable_tss. Aborting."
            )
            raise ValueError

        return out

    @classmethod
    @copy_signature_method(run_bowtie, Self)
    def from_bowtie_split_name(cls, *args: Any, **kwargs: Any):
        return cls.from_sam(run_bowtie(*args, **kwargs), split_name=True)

    @classmethod
    @copy_signature_method(run_bowtie, Self)
    def from_bowtie(cls, *args: Any, **kwargs: Any):
        return cls.from_sam(run_bowtie(*args, **kwargs), split_name=False)


if __name__ == "__main__":
    g = SAMFrame(pl.DataFrame({"name": ["a", "b", "c"], "seq": ["A", "B", "C"]}))
    g.filter("fsd")
