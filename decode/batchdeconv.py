# %%
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from fishtools import progress_bar

PATH = Path("/fast2/3t3clean")

idxs = sorted(
    [int(name.stem.split("-")[1]) for name in PATH.rglob("6_14_22*.tif") if "analysis" not in str(name)]
)
print(len(idxs))
assert len(idxs) == set(idxs).__len__()
# idxs = [i for i in idxs if not (PATH / "down2" / "0" / f"{i:03d}_000.tif").exists()]
# %%
folders = sorted(
    [
        path
        for path in PATH.iterdir()
        if path.is_dir() and path.stem not in ["10x", "analysis", "dapi_edu", "shifts"]
    ]
)
# for i in idxs:
#     if not (PATH / "down2" / "0" / f"{i:03d}_000.tif").exists():
#         print(i)
#         subprocess.run(
#             ["python", "decode/register_prod.py", str(PATH), str(i), "--debug"],
#             capture_output=False,
#             check=True,
#         )
# %%

for folder in folders:
    subprocess.run(
        [
            "python",
            "/home/chaichontat/fishtools/fishtools/analysis/deconv.py",
            "run",
            str(PATH),
            folder.stem.split("--")[0],
            # "--limit",
            # "1000",
        ],
        check=True,
        capture_output=False,
    )
