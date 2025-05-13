from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pyparsing as pp

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class TileConfiguration:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def write(self, path: Path):
        with open(path, "w") as f:
            f.write("dim=2\n")
            for row in self.df.iter_rows(named=True):
                f.write(f"{row['index']:04d}.tif; ; ({row['x']}, {row['y']})\n")

    @classmethod
    def from_pos(cls, df: pd.DataFrame):
        pixel = 2048
        actual = pixel * 0.108
        scaling = 200 / actual
        adjusted = pd.DataFrame(dict(y=(df[0] - df[0].min()), x=(df[1] - df[1].min())))
        adjusted["x"] -= adjusted["x"].min()
        adjusted["x"] *= -(1 / 200) * pixel * scaling
        adjusted["y"] -= adjusted["y"].min()
        adjusted["y"] *= -(1 / 200) * pixel * scaling

        ats = adjusted.copy()
        ats["x"] -= ats["x"].min()
        ats["y"] -= ats["y"].min()

        # mat = ats[["y", "x"]].to_numpy() @ np.loadtxt("/home/chaichontat/fishtools/data/stage_rotation.txt")
        # ats["x"] = mat[:, 0]
        # ats["y"] = mat[:, 1]

        return cls(
            pl.DataFrame(
                ats.reset_index(), schema=pl.Schema({"index": pl.UInt32, "x": pl.Float32, "y": pl.Float32})
            )
        )

    @classmethod
    def from_file(cls, path: str | Path):
        """Parse TileConfiguration.txt file

        Returns:
            pl.DataFrame: ["prefix", "index", "filename", "x", "y"]
        """

        content = Path(path).read_text()

        integer = pp.Word(pp.nums)
        float_number = pp.Combine(
            pp.Optional(pp.oneOf("+ -")) + pp.Word(pp.nums) + "." + pp.Word(pp.nums)
        ).setParseAction(
            lambda t: float(t[0])  # type: ignore
        )

        point = pp.Suppress("(") + float_number("x") + pp.Suppress(",") + float_number("y") + pp.Suppress(")")
        entry = pp.Group(pp.Regex(r"(\w+-)?(\d+)(\.tif)")("filename") + pp.Suppress("; ; ") + point)

        comment = (pp.Literal("#") + pp.restOfLine) | (pp.Literal("dim") + pp.rest_of_line)
        parser = pp.ZeroOrMore(comment.suppress() | entry)

        # Perform this on extracted file name
        # since we want both the raw file name and the index.
        filename = (
            pp.Optional(pp.Word(pp.alphanums) + pp.Literal("-"))
            + integer("index").setParseAction(lambda t: int(t[0]))
            + pp.Literal(".tif")
        )  # type: ignore
        # Parse the content
        out = []
        for x in parser.parseString(content):
            d = x.as_dict()
            out.append(filename.parseString(d["filename"]).as_dict() | d)

        if not len(out):
            raise ValueError(f"File {path} is empty or not in the expected format.")

        return cls(
            pl.DataFrame(
                out,
                schema=pl.Schema({
                    "prefix": pl.Utf8,
                    "index": pl.UInt32,
                    "filename": pl.Utf8,
                    "x": pl.Float32,
                    "y": pl.Float32,
                }),
            )
        )

    def drop(self, idxs: list[int]):
        df = TileConfiguration(self.df.filter(~pl.col("index").is_in(idxs)))
        assert len(df) == len(self) - len(idxs)
        return df

    def downsample(self, factor: int) -> "TileConfiguration":
        if factor == 1:
            return self
        return TileConfiguration(self.df.with_columns(x=pl.col("x") / factor, y=pl.col("y") / factor))

    def __getitem__(self, sl: slice) -> "TileConfiguration":
        return TileConfiguration(self.df[sl])

    def __len__(self) -> int:
        return len(self.df)

    def plot(self, ax: "Axes | None" = None):
        import matplotlib.pyplot as plt

        if not ax:
            _, ax = plt.subplots(figsize=(8, 6), dpi=200)
        for row in self.df.iter_rows(named=True):
            ax.scatter(row["x"], row["y"], s=0.1)
            ax.text(row["x"], row["y"], str(row["index"]), fontsize=6, ha="center", va="center")
        ax.set_aspect("equal")
