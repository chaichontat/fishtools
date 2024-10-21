# %%
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

# PATH = Path("/fast2/3t3clean")
PATH = Path("/mnt/working/e155zachdeconv")


for i in range(0, 6):
    subprocess.run(
        [
            "python",
            Path(__file__).parent / "align_prod.py",
            "optimize",
            str(PATH),
            "--codebook",
            "/home/chaichontat/fishtools/starwork3/zach.json",
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
            "--codebook",
            "/home/chaichontat/fishtools/starwork3/zach.json",
            f"--round={i}",
        ],
        check=True,
        capture_output=False,
    )
