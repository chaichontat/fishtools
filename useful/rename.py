# %%
import re
from pathlib import Path

name = "/mnt/working/e155trcdeconv/registered--left/genestar"
files = list(Path(name).glob("*.pkl"))
for file in files:
    if re.match(r"reg-\w+_\d\d", file.stem):
        file.rename(file.with_stem(file.stem.split("_")[0] + "_genestar_opt" + file.stem.split("_")[1]))

# %%
files = list(Path(name).glob("*.pkl"))
for file in files:
    if re.match(r"reg-\d+_genestar_opt\d\d", file.stem):
        # file.rename(file.with_stem(file.stem.split("_")[0]))
        file.rename(file.with_stem(file.stem.split("_")[0] + "_" + file.stem.split("_")[2]))
# %%
