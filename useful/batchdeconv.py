# %%
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from fishtools import progress_bar

# PATH = Path("/fast2/3t3clean")
PATH = Path(".")
# roi = "full"


# idxs = sorted([int(name.stem.split("-")[1]) for name in PATH.rglob("1_9_17*.tif") if roi in name.parent.name])
# print(len(idxs))
# assert len(idxs) == set(idxs).__len__()
# idxs = [i for i in idxs if not (PATH / "down2" / "0" / f"{i:03d}_000.tif").exists()]
# %%
folders = sorted([
    path
    for path in PATH.iterdir()
    if path.is_dir()
    and not any(
        x in path.name
        for x in [
            "10x",
            "analysis",
            "shifts",
            "fid",
            "registered",
            "old",
            "basic",
            # "1_9_17",
            # "2_10_18",
            # "3_11_19",
            # "4_12_20",
            # "5_13_21",
        ]
    )
])

print("\n".join(map(str, folders)))

for folder in folders:
    subprocess.run(
        [
            "python",
            "/home/chaichontat/fishtools/fishtools/deconv.py",
            # "compute-range",
            # folder,
            "run",
            str(PATH),
            folder.stem.split("--")[0],
            # "--limit",
            # "1000",
        ],
        check=True,
        capture_output=False,
    )
