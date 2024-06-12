# %%
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from fishtools import progress_bar

PATH = Path("/fast2/3t3clean/analysis/deconv/registered")

idxs = sorted([int(name.stem.split("-")[1]) for name in PATH.rglob("reg-*.tif")])
print(len(idxs))
# idxs = [i for i in idxs if not (PATH / "down2" / "0" / f"{i:03d}_000.tif").exists()]
# %%

# for i in idxs:
#     if not (PATH / "down2" / "0" / f"{i:03d}_000.tif").exists():
#         print(i)
#         subprocess.run(
#             ["python", "decode/register_prod.py", str(PATH), str(i), "--debug"],
#             capture_output=False,
#             check=True,
#         )
# %%

with progress_bar(len(idxs)) as callback, ThreadPoolExecutor(8) as exc:
    futs: list[Future] = []
    for i in idxs:
        fut = exc.submit(
            subprocess.run,
            [
                "python",
                Path(__file__).parent / "align_prod.py",
                "run",
                str(PATH / f"reg-{i:04d}.tif"),
                "--scale-file",
                str(PATH / "decode_scale.json"),
            ],
            check=True,
        )
        fut.add_done_callback(callback)

    for x in as_completed(futs):
        x.result()
