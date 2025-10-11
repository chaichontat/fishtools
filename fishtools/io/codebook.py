from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Codebook:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.codebook = json.loads(self.path.read_text())

    @property
    def name(self) -> str:
        return self.path.stem

    def to_dataframe(self, bits_as_list: bool = False):
        """
        Return a Polars DataFrame view of this codebook.

        Columns
        - target (Utf8): Gene/target name in the codebook.
        - bit0, bit1, bit2 (UInt8): When ``bits_as_list=False`` (default), three
          integer bit/channel identifiers indicating which channels define the
          target. The project’s codebooks encode up to three bits per target.
        - bits (List[UInt8]): When ``bits_as_list=True``, a single list column
          containing the bit identifiers for the target; replaces ``bit0..2``.
        """
        import polars as pl

        df = (pl.DataFrame(self.codebook).transpose(include_header=True)).rename({"column": "target"})

        if not bits_as_list:
            return df.rename({"column_0": "bit0", "column_1": "bit1", "column_2": "bit2"})

        return df.with_columns(
            concat_list=pl.concat_list("column_0", "column_1", "column_2")
            .cast(pl.List(pl.UInt8))
            .alias("bits")
        )


__all__ = ["Codebook"]
