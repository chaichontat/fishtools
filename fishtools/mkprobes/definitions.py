from typing import Protocol

import polars as pl


class Filter(Protocol):
    def __call__(self, fasta: str) -> pl.DataFrame: ...
