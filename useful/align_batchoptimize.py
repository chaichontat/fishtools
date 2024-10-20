# %%
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

# PATH = Path("/fast2/3t3clean")
PATH = Path("/mnt/working/e155trcdeconv/registered--left")


for i in range(2, 10):
    subprocess.run(
        [
            "python",
            Path(__file__).parent / "align_prod.py",
            "optimize",
            str(PATH),
            "--batch-size=50",
            f"--round={i}",
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
            f"--round={i}",
        ],
        check=True,
        capture_output=False,
    )
