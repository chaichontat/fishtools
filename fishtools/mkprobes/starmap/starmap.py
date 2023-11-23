from dataclasses import dataclass

import polars as pl
import primer3

from fishtools import hp, rc, tm


def rotate(s: str, r: int):
    return s[r:] + s[:r]


def test_splint_padlock(splint: str, padlock: str, lengths: tuple[int, int] = (6, 6)):
    # if not len(splint) & 1 == 0:
    #     raise ValueError("Splint must be even length")
    # mid = len(splint) // 2
    splint, padlock = splint.upper(), padlock.upper()
    start, end = lengths

    splint = splint[-(start + end) :]
    ok_start = rc(splint[:start]) == padlock[:start]
    ok_end = rc(splint[start:]) == padlock[-end:]
    return ok_start and ok_end


def split_probe(seq: str, target_tm: float) -> tuple[str, str, str] | None:
    first, last = "", ""
    i = 0
    for start in range(5):
        for i in range(start + 18, start + 28):
            if tm(first := seq[start:i], "hybrid") - 5 > target_tm:
                break

        for offset in reversed(range(3)):  # 2, 1, 0
            for j in range(18, 28):
                if i + offset + j >= len(seq):
                    break

                if tm(last := seq[i + offset : i + offset + j], "hybrid") - 5 > target_tm:
                    assert len(last) > 0
                    if hp(first, "dna") < target_tm and hp(last, "dna") < target_tm:
                        # bc of polars datatype
                        return first, last, str(i + offset)

    return "", "", str(-1)


def pad(s: str, target: int = 89):
    assert len(s) <= target
    return s + "ATAGTTATAT"[: target - len(s)]


def gen_rotate(dfs: pl.DataFrame):
    res = dfs.with_columns(
        rotated=pl.col("seq")
        .apply(rc)
        .apply(pad)
        .apply(lambda x: rotate(x, 20 - 6 - 3), return_dtype=pl.Utf8)
    ).with_columns(
        splint_="TGTTGATGAGGTGTTGATGAT"
        + "AA"
        + pl.col("splint").apply(rc)
        + "ATA"  # mismatch
        + pl.col("rotated").str.slice(0, 6).apply(rc)
        + pl.col("rotated").str.slice(-6, 6).apply(rc)
    )

    # uyu = res["splint_"].apply(lambda x: BsaI.catalyze(Seq.Seq(x)))
    # uyu.filter(uyu.list.lengths() != 1)

    assert (res["splint_"].apply(lambda x: BsaI.search(Seq.Seq(x))).list.lengths() == 0).all()


# def gen_3splint(
#     splited: list[tuple[str, str]],
#     bits: list[str],
#     splint: str = "GTAAGGTGAATA",
#     primer_gap: str = "CA",
#     pad_gap: str = "TA",
# ):
#     splints = (rc(splint[: len(splint) // 2]), rc(splint[len(splint) // 2 :]))
#     footer = primer_gap + rc(bits[0][-11:-1])
#     primer = [s[0] + footer for s in splited]
#     pad = [
#         splints[0]
#         + bits[0].lower()
#         + pad_gap
#         + s[1]
#         + bits[1].lower()
#         + "TATTAAAT"[: 79 - len(s[1]) - 54]
#         + splints[1]
#         for s in splited
#     ]
#     assert all(test_splint_padlock(splint, s) for s in pad)
#     assert all(0 < s.index(rc(footer[2:]).lower()) < len(s) // 2 for s in pad)
#     return primer, pad


# def gen_starpad(
#     splits: list[tuple[str, str]],
#     bits: list[str],
#     pad_gap: str = "ta",
#     spacer: str = "TATTCAAT",
#     target_length: int = 67,
#     n_bits: int = 2,
#     constant: str = "",
# ):
#     """
#     Splits must be reverse complemented.
#     Split[0] binds toward the 3' end of the RNA, will become splint.
#     Split[1] binds toward the 5' end of the RNA, will become padlock.
#     """
#     combi = cycle(combinations(bits, n_bits))

#     def inner(s: tuple[str, str]):
#         bbs = list(next(combi))

#         draft = (
#             pad_gap
#             + s[1]
#             + "AA".join([constant] if constant else [] + bbs)
#             + spacer[: target_length - len(s[1]) - 42]
#         )

#         if len(draft) > target_length:
#             raise ValueError(f"Generated padlock is too long: {len(draft)}")

#         out_pad = rotate(draft, -6)
#         splint = s[0] + (footer := pad_gap + rc(out_pad[:6]) + rc(out_pad[-6:]))

#         assert test_splint_padlock(footer[2:], out_pad), out_pad
#         assert splits[1] == out_pad[6 + len(pad_gap) : 6 + len(pad_gap) + len(splits[1])]
#         return splint, out_pad

#     res = [inner(s) for s in splits]
#     return [s[0] for s in res], [s[1] for s in res]


@dataclass
class STARPrimers:
    header: str
    bsa_hang: str
    footer: str
    is_splint: bool = False

    @staticmethod
    def check_primer(seq: str):
        if not primer3.calc_homodimer_tm(seq) < 40:
            raise ValueError(primer3.calc_homodimer_tm(seq))
        if not primer3.calc_hairpin_tm(seq) < 40:
            raise ValueError(primer3.calc_hairpin_tm(seq))

    def __post_init__(self):
        if not len(self.bsa_hang) == 1:
            raise ValueError(f"BsaI overhang must be 1 nt long. Got {self.bsa_hang}.")
        if not len(self.footer) == 5:
            raise ValueError(f"Footer must be 5 nt long. Got {self.footer}.")
        self.check_primer(self.header)

    def make_header(self):
        return self.header + "GGTCTC" + self.bsa_hang

    def gen_cleave1(self):
        return (
            ("DDDDDDDD" if not self.is_splint else "CATCAACA")
            + rc(self.bsa_hang)
            + rc("GGTCTC")
            + rc(self.header)[:4]
        )

    def make_footer(self):
        return self.footer + "GAGACC" + "ATTATCCCTATAGTGAG"

    def gen_cleave2(self):
        return rc(self.make_footer())[-(10 + 5) :] + ("HHHHH" if self.is_splint else "DDDDD")
