# %%
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from tqdm import tqdm


@click.command()
@click.argument("folder", type=click.Path(exists=True))
def main(folder: str):
    with ThreadPoolExecutor(64) as executor:
        list(
            tqdm(
                executor.map(
                    subprocess.run,
                    [
                        ["python", "tojp2.py", file.as_posix()]
                        for file in Path(folder).glob("*.dax")
                    ],
                )
            )
        )


if __name__ == "__main__":
    main()
