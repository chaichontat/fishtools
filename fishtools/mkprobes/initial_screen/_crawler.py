import polars as pl
from oligocheck.seqcalc import Model, hp_fish, tm
from oligocheck.sequtils import gc_content


def crawler(
    seq: str,
    prefix: str,
    length_limit: tuple[int, int] = (25, 46),
    gc_limit: tuple[float, float] = (0.3, 0.7),
    tm_limit: float = 51,
    hairpin_limit: float = 40 + 0.65 * 30,
    tm_model: Model = "rna",
    to_avoid: list[str] | None = None,
) -> pl.DataFrame:
    """
    Based on the monotonic relationship between Tm and sequence length.

    1. Length Check:
        If the subsequence length is outside `length_limit`, the subsequence is lengthened.
    2. GC Content Check:
        If the GC content is outside `gc_limit`, the subsequence is lengthened and the function loops back to step 1.
    3. Tm Check:
        If the Tm exceeds `tm_limit`, the function proceeds to the hairpin structure check.
        If not, it lengthens the subsequence and loops back to step 1.
    4. Hairpin Structure Check:
        If the hairpin structure value is below `hairpin_limit`, the subsequence is added to the output.
    5. Backtracking:
        After checking Tm and hairpin structure, or if the length exceeds the upper `length_limit`,
        the function moves to the next start position in the sequence and restarts from step 1.

    The function excludes subsequences with homopolymer runs of A's, T's, C's, or G's in the final DataFrame.
    """

    end = length_limit[0]
    names = []
    seqs = []
    if to_avoid is None:
        to_avoid = []
    to_avoid = to_avoid.copy()
    to_avoid.extend(["A" * 5, "T" * 5, "C" * 5, "G" * 5])

    for start in range(len(seq) - length_limit[0]):
        if seq[start].lower() == "n":
            continue

        while True:
            if end - start < length_limit[0]:
                end += 1
                continue
            if end - start > length_limit[1] or end > len(seq):
                break
            if (gc_content(seq[start:end]) < gc_limit[0]) or (gc_content(seq[start:end]) > gc_limit[1]):
                end += 1
                continue

            if tm(seq[start:end], model=tm_model) > tm_limit:
                if hp_fish(seq[start:end]) < hairpin_limit:
                    names.append(f"{prefix}:{start}-{end-1}")
                    seqs.append(seq[start:end])
                break
            end += 1
    return pl.DataFrame(dict(name=names, seq=seqs)).filter(~pl.col("seq").str.contains("|".join(to_avoid)))
