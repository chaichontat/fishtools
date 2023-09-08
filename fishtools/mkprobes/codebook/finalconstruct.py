from pathlib import Path

import click


# fmt: off
@click.command()
@click.argument("output_path", type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--readouts", "-r", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--codebook", "-c", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.option("--head", type=str, help="Sequence to add to the 5' end", default="")
@click.option("--tail", type=str, help="Sequence to add to the 3' end", default ="")
@click.option("--rc", is_flag=True, help="Reverse complement (for DNA probes which needs RT)")
@click.option("--maxn", type=int, help="Maximum number of probes per gene", default=64)
@click.option("--minn", type=int, help="Minimum number of probes per gene", default=48)
# fmt: on
def construct(
    output_path: Path,
    readouts: Path,
    codebook: Path,
    head: str = "",
    tail: str = "",
    rc: bool = True,
    maxn: int = 64,
    minn: int = 48,
):
    ...
