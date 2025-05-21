import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

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

pair_name = pl.col("name").str.extract(r"^(.*)_(splint|padlock)$", 1)


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
    stats = {}
    selected_global = set()
    # tss = df["transcript_ori"].unique().to_list()
    # assert len(tss) == 1, df

    # for ts in tss:
    logger.info(f"Number of candidates that match any filter: {len(df)}/{len(ddf)}")
    stats["start"] = len(df)
    stats["match_any"] = len(df)

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
            stats[f"selected_{i}"] = len(sel_local)
            if len(sel_local) > n:
                break
        except RecursionError:
            print("Recursion error")
            break

    selected_global |= sel_local  # type: ignore
    selected_names = df.filter(pl.col("index").is_in(selected_global)).select(name=pair_name)["name"]
    df = df.filter(pair_name.is_in(selected_names))
    logger.info(f"Selected {len(df) // 2} probes.")
    stats["selected_pair"] = len(df) // 2
    return df, stats


def the_filter(
    df: pl.DataFrame,
    overlap: int = -1,
    max_tm_offtarget: float = 30,
    max_hp: float = 37,
) -> tuple[pl.DataFrame, dict]:
    ff, stats = handle_overlap(
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
    )
    return ff.sort("priority"), stats


def visualize_probe_coverage(
    pos_starts: Sequence[int],
    pos_ends: Sequence[int],
    gene_length: int,
    label: str = "Probe Coverage",
    thresholds: list[float | str] | None = None,
    output_file: str | Path | None = None,
    max_line_width: int = 110,
):
    """
    Generates an ASCII visualization of probe coverage along a gene.
    If gene_length exceeds max_line_width, the plot is scaled down.

    Args:
        pos_starts: A sequence of 1-indexed, inclusive start positions for probes.
        pos_ends: A sequence of 1-indexed, inclusive end positions for probes.
        gene_length: The total length of the gene (1-indexed).
        label: A label for the visualization block.
        thresholds: A list of percentage thresholds (e.g., [0.1, 0.25, "any"])
                    for which to generate coverage lines. "any" means any coverage.
                    Defaults to ["any", 0.1, 0.25, 0.5, 0.75].
        output_file: Optional path to a file to save the visualization.
                     If None, prints to stdout.
        max_line_width: Maximum width for each line in the output. The plot
                        itself will be scaled if gene_length exceeds available plot space.
    """
    output_lines = []
    prefix_width = 4  # For labels like "any ", "10% "
    content_plot_width = max(1, max_line_width - prefix_width)

    def _add_line_to_output(line: str):
        output_lines.append(line)

    def _add_wrapped_line(prefix_content: str, main_content: str):
        current_prefix = f"{prefix_content:<{prefix_width}}"
        if not main_content:
            _add_line_to_output(current_prefix.rstrip())
            return

        if len(main_content) <= content_plot_width:
            _add_line_to_output(f"{current_prefix}{main_content}")
        else:
            # This branch should ideally only be hit for long titles/labels,
            # not for the plot data if pre-scaled correctly.
            _add_line_to_output(f"{current_prefix}{main_content[:content_plot_width]}")
            remaining_str = main_content[content_plot_width:]
            while remaining_str:
                chunk = remaining_str[:content_plot_width]
                _add_line_to_output(f"{'':<{prefix_width}}{chunk}")
                remaining_str = remaining_str[content_plot_width:]

    if gene_length <= 0:
        _add_line_to_output(f"\n--- {label} ---")
        _add_line_to_output(f"Invalid gene length: {gene_length}")
        if output_file:
            Path(output_file).write_text("\n".join(output_lines) + "\n")
        else:
            for line in output_lines:
                print(line)
        return

    gene_length_int = int(gene_length)
    effective_thresholds = thresholds if thresholds is not None else ["any", 0.1, 0.25, 0.5, 0.75]

    _add_line_to_output(f"\n--- {label} (Gene Length: {gene_length_int}) Total: {len(pos_starts)} pairs ---")

    # Determine the actual width for the plot string and if scaling is needed
    is_scaled: bool
    current_plot_string_width: int
    if gene_length_int <= content_plot_width:
        current_plot_string_width = gene_length_int
        is_scaled = False
    else:
        current_plot_string_width = content_plot_width
        is_scaled = True

    # 1. First line: Gene boundary
    header_line_content: str
    if not is_scaled:
        if current_plot_string_width == 1:  # gene_length_int is 1
            header_line_content = "1"
        elif current_plot_string_width == 2:  # gene_length_int is 2
            header_line_content = "12"
        else:  # gene_length_int > 2
            header_line_content = f"1{'-' * (current_plot_string_width - 2)}{gene_length_int}"
    else:  # Scaled header
        end_label = str(gene_length_int)
        if current_plot_string_width == 1:
            header_line_content = "*"  # Indicates entire scaled gene
        elif current_plot_string_width == 2:
            if len(end_label) == 1:
                header_line_content = f"1{end_label}"
            else:
                header_line_content = "1~"  # Approx end
        else:  # current_plot_string_width > 2
            padding_len = current_plot_string_width - 1 - len(end_label)
            if padding_len >= 0:
                header_line_content = f"1{'-' * padding_len}{end_label}"
            else:  # Not enough space for '1', dashes, and full end_label
                header_line_content = f"1{'-' * (current_plot_string_width - 1)}"
    _add_wrapped_line(" ", header_line_content)

    if not len(pos_starts) or not len(pos_ends) or len(pos_starts) != len(pos_ends):
        _add_line_to_output("No probes or mismatched start/end positions to visualize.")
        for t_val in effective_thresholds:
            t_s = f"{t_val * 100:.0f}%" if isinstance(t_val, (float, int)) else str(t_val)
            # For empty probes, the plot line is just spaces of the determined width
            _add_wrapped_line(t_s, " " * current_plot_string_width)
        if output_file:
            Path(output_file).write_text("\n".join(output_lines) + "\n")
        else:
            for line in output_lines:
                print(line)
        return

    coverage_array = np.zeros(gene_length_int, dtype=int)
    for start, end in zip(pos_starts, pos_ends):
        start_idx = int(start) - 1
        end_idx = int(end) - 1
        start_idx = max(0, min(start_idx, gene_length_int - 1))
        end_idx = max(0, min(end_idx, gene_length_int - 1))
        if start_idx <= end_idx:
            coverage_array[start_idx : end_idx + 1] += 1
    total_probes = len(pos_starts)

    # 3. Generate lines for each threshold
    for threshold_val in effective_thresholds:
        line_chars = [" "] * current_plot_string_width

        if not is_scaled:
            for i in range(current_plot_string_width):  # which is gene_length_int
                current_coverage_at_pos_i = coverage_array[i]
                if threshold_val == "any":
                    if current_coverage_at_pos_i > 0:
                        line_chars[i] = "-"
                elif isinstance(threshold_val, (float, int)):
                    if total_probes > 0 and (current_coverage_at_pos_i / total_probes) > threshold_val:
                        line_chars[i] = "-"
        else:  # Scaled plot line
            for j in range(current_plot_string_width):  # j is display character index
                orig_start_idx = math.floor(j * gene_length_int / current_plot_string_width)
                orig_next_start_idx = math.floor((j + 1) * gene_length_int / current_plot_string_width)

                # Segment of original coverage array for this display character
                segment_coverage_values = coverage_array[orig_start_idx:orig_next_start_idx]

                if segment_coverage_values.size > 0:
                    if threshold_val == "any":
                        if np.any(segment_coverage_values > 0):
                            line_chars[j] = "-"
                    elif isinstance(threshold_val, (float, int)):
                        if total_probes > 0 and np.any(
                            (segment_coverage_values / total_probes) > threshold_val
                        ):
                            line_chars[j] = "-"

        threshold_str_display = (
            f"{threshold_val * 100:.0f}%" if isinstance(threshold_val, (float, int)) else str(threshold_val)
        )
        _add_wrapped_line(threshold_str_display, "".join(line_chars))

    if output_file:
        Path(output_file).write_text("\n".join(output_lines) + "\n")
    else:
        for ol in output_lines:
            print(ol)
