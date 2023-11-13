from itertools import combinations, cycle

from fishtools import rc, tm


def rotate(s: str, r: int):
    return s[r:] + s[:r]


def test_splint_padlock(splint: str, padlock: str, length: int = 6):
    # if not len(splint) & 1 == 0:
    #     raise ValueError("Splint must be even length")
    # mid = len(splint) // 2
    splint = splint[-(length * 2) :]
    return rc(splint[:length]) == padlock[:length] and rc(splint[length:]) == padlock[-length:]


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
                    return first, last, str(i + offset)  # bc of polars datatype

    return "", "", str(-1)


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


# def gen_phiamp():
