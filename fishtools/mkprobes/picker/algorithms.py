from bisect import bisect_left
from typing import Sequence

import numpy as np


class OverlapWeighted:
    def __init__(self, start: Sequence[int], end: Sequence[int], priorities: Sequence[int], overlap: int = 0):
        self.start = start
        self.end = end
        self.priorities = priorities
        self.overlap = overlap
        self.n = len(start)

        if not (len(start) == len(end) == len(priorities)):
            raise ValueError("Lengths not equal")
        if any(end[i + 1] < end[i] for i in range(self.n - 1)):
            raise ValueError("Ends not sorted")
        self.opt = np.ones(self.n + 1, dtype=np.uint32) * -1
        self.opt[0] = 0
        self.res = []

    def q(self, j: int) -> int:
        """Largest index i < j such that job i is compatible with j"""
        # bisect_left
        # The return value i is such that all e in a[:i] have e < x, and all e in a[i:] have e >= x.
        # So if x already appears in the list, a.insert(i, x) will insert just before the leftmost x already there.
        # Guarantees that overlap = 0 means no overlap
        return bisect_left(self.end[: j - 1], self.start[j - 1] + self.overlap)

    def dp(self, j: int) -> int:
        if self.opt[j] == -1:
            self.opt[j] = max(self.priorities[j - 1] + self.dp(self.q(j)), self.dp(j - 1))
        return self.opt[j]

    def backtrack(self, j: int) -> None:
        if j == 0:
            return
        if self.priorities[j - 1] + self.dp(self.q(j)) > self.opt[j - 1]:
            self.res.append(j - 1)
            self.backtrack(self.q(j))
        else:
            self.backtrack(j - 1)

    def run(self) -> list[int]:
        self.dp(self.n)
        self.backtrack(self.n)
        return sorted(self.res)


# weighted activity selection algorithm
def find_overlap_weighted(
    start: Sequence[int], end: Sequence[int], priorities: Sequence[int], overlap: int = 0
) -> list[int]:
    return OverlapWeighted(start, end, priorities, overlap).run()


# greedy activity selection
def find_overlap(start: Sequence[int], end: Sequence[int], overlap: int = 0) -> list[int]:
    out = [0]
    curr_end = end[0]
    for i in range(1, len(start)):
        if end[i] < curr_end:
            raise ValueError("Ends not sorted")
        if start[i] - overlap > end[out[-1]]:
            out.append(i)
    return out
