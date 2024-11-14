from typing import Final, Sequence, cast

import numpy as np
import polars as pl
from loguru import logger

from ._algorithms import find_overlap, find_overlap_weighted

# fmt: off
PROBE_CRITERIA: Final = dict(
    ok_quad_c = ~pl.col("seq").str.contains("GGGG"),
    ok_quad_a = ~pl.col("seq").str.contains("TTTT"),
    # ok_stack_c= pl.all([pl.col("seq").str.slice(-6 - i, 6).str.count_matches("G").lt(4) for i in range(6)]),
    ok_stack_c= pl.fold(True, (lambda acc, x: acc & x), [pl.col("seq").str.slice(-6 - i, 6).str.count_matches("G").lt(4) for i in range(6)]),
    ok_comp_a =(pl.col("seq").str.count_matches("T") / pl.col("seq").str.len_chars() < 0.28),
    gc_content=(pl.col("seq").str.count_matches("G|C") / (pl.col("seq").str.len_chars()))
)
# fmt: on

pair_name = pl.col("name").str.split("_").list.get(1)


def filter_have_both(df: pl.DataFrame):
    have_both = df.group_by(pair_name).count().filter(pl.col("count") == 2)
    return df.filter(pair_name.is_in(have_both["name"]))


def handle_overlap(
    df: pl.DataFrame,
    criteria: list[pl.Expr],
    overlap: int = -2,
    n: int = 100,
):
    if len(gene := df.select(pl.col("gene").unique())) > 1:
        raise ValueError("More than one gene in filtered")
    gene = gene.item()

    df = df.sort(by=["pos_end", "tm"], descending=[False, True])
    criteria = criteria or [pl.col("*")]

    df_ori = df.with_row_count("index")

    ddf = df_ori.lazy().with_columns(priority=pl.lit(0, dtype=pl.UInt8))
    for priority, criterion in reversed(list(enumerate(criteria, 1))):
        ddf = ddf.update(
            ddf.filter(criterion).with_columns(priority=pl.lit(priority, dtype=pl.UInt8)),
            on="index",
        )
    ddf = ddf.collect()
    df = ddf.filter(pl.col("priority") > 0)
    df = filter_have_both(df)

    # breakpoint()

    selected_global = set()
    # tss = df["transcript_ori"].unique().to_list()
    # assert len(tss) == 1, df

    # for ts in tss:
    logger.info(f"Number of candidates that match any filter: {len(df)}/{len(ddf)}")
    if not len(df):
        logger.critical("No probes passed filters")
        exit(0)
    logger.info(f"Max pos_end: {df['pos_end'].max()}")
    for i in range(1, len(criteria) + 1):
        run = (
            df.filter((pl.col("priority") <= i) & ~pl.col("index").is_in(selected_global))
            .filter(pl.col("name").str.ends_with("splint"))
            .select(["index", "pos_start", "pos_end", "priority"])
            .sort(["pos_end", "pos_start"])
        )
        if not len(run):
            continue

        priorities = np.sqrt(len(criteria) + 1 - run["priority"])
        try:
            if i == 1:
                ols = find_overlap(
                    cast(Sequence[int], run["pos_start"]),
                    cast(Sequence[int], run["pos_end"]),
                    overlap=overlap,
                )
            else:
                ols = find_overlap_weighted(
                    cast(Sequence[int], run["pos_start"]),
                    cast(Sequence[int], run["pos_end"]),
                    cast(Sequence[int], priorities),
                    overlap=overlap,
                )
            sel_local = set(run[ols]["index"].to_list())
            logger.info(f"Priority {i}, selected {len(sel_local)} probes")
            if len(sel_local) > n:
                break
        except RecursionError:
            print("Recursion error")
            break

    selected_global |= sel_local  # type: ignore
    selected_names = df.filter(pl.col("index").is_in(selected_global)).select(name=pair_name)["name"]
    df = df.filter(pair_name.is_in(selected_names))
    logger.info(f"Selected {len(df) // 2} probes.")
    return df


def the_filter(
    df: pl.DataFrame, overlap: int = -1, max_tm_offtarget: float = 30, max_hp: float = 37
) -> pl.DataFrame:
    return handle_overlap(
        df,
        criteria=[
            # fmt: off
            (pl.col("oks") > 4)
            & (pl.col("hp") < max_hp)
            & pl.col("max_tm_offtarget").lt(max_tm_offtarget)
            & (pl.col("maps_to_pseudo").is_null() | pl.col("maps_to_pseudo").eq("")),
            (pl.col("oks") > 3)
            & (pl.col("hp") < max_hp)
            & pl.col("max_tm_offtarget").lt(max_tm_offtarget)
            & (pl.col("maps_to_pseudo").is_null() | pl.col("maps_to_pseudo").eq("")),
            (pl.col("oks") > 3) & (pl.col("hp") < max_hp) & pl.col("max_tm_offtarget").lt(max_tm_offtarget),
            (pl.col("oks") > 3)
            & (pl.col("hp") < max_hp + 5)
            & pl.col("max_tm_offtarget").lt(max_tm_offtarget + 4),
            (pl.col("oks") > 2)
            & (pl.col("hp") < max_hp + 5)
            & pl.col("max_tm_offtarget").lt(max_tm_offtarget + 4),
            (pl.col("oks") > 1)
            & (pl.col("hp") < max_hp + 5)
            & pl.col("max_tm_offtarget").lt(max_tm_offtarget + 4),
            # fmt: on
        ],
        overlap=overlap,
    ).sort("priority")
