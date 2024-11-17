# %%
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from fishtools import progress_bar

PATH = Path("/mnt/archive/starmap/zne172/20241113-ZNE172-Zach/analysis/deconv")
roi = "right"
chan = "4_12_20"
idxs = sorted({int(name.stem.split("-")[1]) for name in PATH.rglob(f"{chan}--{roi}/{chan}*.tif")})
print(len(idxs))

if not PATH.exists():
    raise ValueError(f"Path {PATH} does not exist.")
# idxs = [i for i in idxs if not (PATH / "down2" / "0" / f"{i:03d}_000.tif").exists()]
# %%

# %%

with progress_bar(len(idxs)) as callback, ThreadPoolExecutor(4) as exc:
    futs: list[Future] = []
    for i in idxs:
        fut = exc.submit(
            subprocess.run,
            [
                "python",
                Path(__file__).parent.parent / "useful" / "register_prod.py",
                str(PATH),
                str(i),
                "--fwhm=4",
                "--threshold=4",
                "--reference",
                chan,
                f"--roi={roi}",
                # "--overwrite",
            ],
            check=True,
        )
        fut.add_done_callback(callback)

    for x in as_completed(futs):
        x.result()
