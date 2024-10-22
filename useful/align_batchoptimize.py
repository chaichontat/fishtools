# %%
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

# PATH = Path("/fast2/3t3clean")
PATH = Path("/mnt/working/e155trcdeconv")
codebook = "/home/chaichontat/fishtools/starwork3/ordered/genestar.json"


for i in range(0, 10):
    logger.critical(f"Starting round {i}")

    subprocess.run(
        [
            "python",
            Path(__file__).parent / "align_prod.py",
            "optimize",
            str(PATH),
            "--codebook",
            codebook,
            "--batch-size=50",
            f"--round={i}",
            "--threads=16",
            "--overwrite",
        ],
        check=True,
        capture_output=False,
    )
    subprocess.run(
        [
            "python",
            Path(__file__).parent / "align_prod.py",
            "combine",
            str(PATH),
            "--codebook",
            codebook,
            f"--round={i}",
        ],
        check=True,
        capture_output=False,
    )
