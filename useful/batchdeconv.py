# %%
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from fishtools import progress_bar

# PATH = Path("/fast2/3t3clean")
PATH = Path("/mnt/archive/starmap/at8_95")
roi = "full"


idxs = sorted([
    int(name.stem.split("-")[1]) for name in PATH.rglob("4_12_20*.tif") if roi in name.parent.name
])
print(len(idxs))
assert len(idxs) == set(idxs).__len__()
# idxs = [i for i in idxs if not (PATH / "down2" / "0" / f"{i:03d}_000.tif").exists()]
# %%
folders = sorted([
    path
    for path in PATH.iterdir()
    if path.is_dir() and path.stem not in ["10x", "analysis", "dapi_edu", "shifts"]
])

for folder in folders:
    subprocess.run(
        [
            "python",
            "/home/chaichontat/fishtools/fishtools/analysis/deconv.py",
            "run",
            # "compute-range",
            # folder,
            # str(PATH),
            # folder.stem.split("--")[0],
            # "--limit",
            # "1000",
        ],
        check=True,
        capture_output=False,
    )
