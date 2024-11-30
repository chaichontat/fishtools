# %%
import subprocess
from pathlib import Path

from fishtools.utils.pretty_print import progress_bar_threadpool

PATH = Path("/mnt/archive/starmap/zne172/20241119-ZNE172-Trc/analysis/deconv")
rois = ["centercortex"]  # ["rightcortex", "rightmid", "leftcortex", "leftmid", "centermid", "centercortex"]
chan = "4_12_20"

idxs = None
if not PATH.exists():
    raise ValueError(f"Path {PATH} does not exist.")
use_custom_idx = idxs is not None
# idxs = [i for i in idxs if not (PATH / "down2" / "0" / f"{i:03d}_000.tif").exists()]
# %%

# %%

for roi in rois:
    if not use_custom_idx:
        idxs = sorted({int(name.stem.split("-")[1]) for name in PATH.rglob(f"{chan}--{roi}/{chan}*.tif")})
        print(len(idxs))

    assert idxs
    with progress_bar_threadpool(len(idxs), threads=4) as submit:
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
