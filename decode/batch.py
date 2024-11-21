# %%
import subprocess
from pathlib import Path

from fishtools.utils.pretty_print import progress_bar_threadpool

PATH = Path("/mnt/archive/starmap/zne172/20241113-ZNE172-Zach/analysis/deconv")
roi = "right"
chan = "4_12_20"
assert PATH.exists()
idxs = sorted({int(name.stem.split("-")[1]) for name in PATH.rglob(f"{chan}--{roi}/{chan}*.tif")})
print(len(idxs))

if not PATH.exists():
    raise ValueError(f"Path {PATH} does not exist.")
# idxs = [i for i in idxs if not (PATH / "down2" / "0" / f"{i:03d}_000.tif").exists()]
# %%

# %%

with progress_bar_threadpool(len(idxs), threads=10) as submit:
    for i in idxs:
        fut = submit(
            subprocess.run,
            [
                "python",
                Path(__file__).parent.parent / "useful" / "register_prod.py",
                str(PATH),
                str(i),
                "--fwhm=5",
                "--threshold=5",
                "--reference",
                chan,
                f"--roi={roi}",
                # "--overwrite",
            ],
            # check=True,
        )
