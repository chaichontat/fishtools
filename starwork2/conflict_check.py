import json
from itertools import chain
from pathlib import Path
from typing import Collection, Hashable, TypeVar

import rich_click as click
from loguru import logger

T = TypeVar("T", bound=Hashable)


def check_dupes(xs: Collection[Collection[T]]) -> set[T]:
    if not xs or not all(map(len, xs)):
        raise ValueError("Some elements empty.")

    flat = set(chain.from_iterable(xs))
    if len(set(map(type, flat))) > 1:
        raise Exception("Collections contain different types of elements")

    for li, se in zip(xs, sets := list(map(set, xs))):
        if len(li) != len(se):
            seen, dupe = set(), []
            for x in li:
                if x in seen:
                    dupe.append(x)
                seen.add(x)
            raise ValueError(f"Duplicates presented in one of the collections. {dupe}")

    if overlaps := set.intersection(*sets):
        raise Exception(f"Overlaps: {list(overlaps)}")

    # Return the unique elements
    return set(flat)


@logger.catch
@click.command()
@click.argument(
    "codebooks", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path)
)
def main(codebooks: list[Path]):
    js: list[dict[str, list[int]]] = [json.loads(c.read_text()) for c in codebooks]
    genes = [list("-".join(ts.split("-")[:-1]) for ts in cb) for cb in js]

    gs = check_dupes(genes)
    logger.info(f"{len(gs)} genes total in {codebooks}.")

    bits = [set(chain.from_iterable(cb.values())) for cb in js]
    bs = check_dupes(bits)
    logger.info(f"{len(bs)} bits total: {bs}")


if __name__ == "__main__":
    main()
