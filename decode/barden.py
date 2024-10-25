# %%
import json
from pathlib import Path

import bardensr
import bardensr.plotting
import IPython.display
import matplotlib.pylab as plt
import numpy as np
import tifffile

# %%
n_chans = 20
cb = json.loads(Path("starwork3/genestar.json").read_text())
arr = np.zeros((len(cb) - 1, n_chans), dtype=bool)
for i, v in enumerate(cb.values()):
    if 28 in v:
        continue
    for a in v:
        assert a > 0
        arr[i, a - 1] = 1

# %%
img = tifffile.imread("trc.tif")[:20, 24:2024, 24:2024].reshape(20, -1, 2000, 2000) + 1
# %%
bench = bardensr.load_example("ab701a5a-2dc3-11eb-9890-0242ac110002")
R, C, J = bench.codebook.shape
F = R * C
Xflat = bench.X.reshape((R * C,) + bench.X.shape[-3:])
codeflat = bench.codebook.reshape((28, -1))
# %%
Xflat = (img - np.min(img, axis=(1, 2, 3)).reshape(20, 1, 1, 1) + 1) / np.mean(img, axis=(1, 2, 3)).reshape(
    20, 1, 1, 1
)

Xnorm = bardensr.preprocessing.background_subtraction(Xflat, [0, 10, 10])
# Xnorm = bardensr.preprocessing.minmax(Xnorm)
# %%
Xnorm = (Xnorm - np.min(Xnorm, axis=(1, 2, 3)).reshape(20, 1, 1, 1) + 1) / np.mean(
    Xnorm, axis=(1, 2, 3)
).reshape(20, 1, 1, 1)

# %%

evidence_tensor = bardensr.spot_calling.estimate_density_singleshot(Xnorm, arr.T, noisefloor=0.05)
# %%
with bardensr.plotting.AnimAcross(sz=2) as a:
    for j in range(5):
        a(f"barcode {j}")
        plt.imshow(evidence_tensor[0, :, :, j], vmin=0, vmax=1)
        plt.axis("off")
# %%
thresh = 0.4
result = bardensr.spot_calling.find_peaks(evidence_tensor, thresh)
result
# %%


# %%
evidence_tensor_iterative, extra_learned_params = bardensr.spot_calling.estimate_density_iterative(
    Xnorm[:, :, 500:1000, 500:1000], arr.T, iterations=100
)


# %%
plt.imshow(extra_learned_params["frame_gains"].reshape((5, 4)).T, vmin=0)
plt.colorbar()
plt.xlabel("rounds")
plt.ylabel("channels")


# %%
thresh_iterative = evidence_tensor_iterative.max() * 0.15
result_iterative = bardensr.spot_calling.find_peaks(evidence_tensor_iterative, thresh_iterative)
print(len(result_iterative))
# %%
# %%
i = 600
m1 = result.loc[i, "m1"]
m2 = result.loc[i, "m2"]
j = result.loc[i, "j"]


def go(r, c, *args):
    plt.imshow(Xnorm[r * 4 + c, 0, m1 - 10 : m1 + 10, m2 - 10 : m2 + 10])
    if arr[j, r * 4 + c]:
        plt.axhline(10, color="red")
        plt.axvline(10, color="red")


bardensr.plotting.plot_rbc(5, 4, go, sz=1, sideways=True, notick=True)
# %%
out = np.zeros((500, 500))
for row in result_iterative.itertuples():
    out[row.m2, row.m1] = row.j
plt.imshow(out)
# %%
plt.imshow(Xnorm[:, 0, 500:1000, 500:1000].mean(axis=0), cmap="gray", vmax=60)
plt.scatter(result_iterative["m2"], result_iterative["m1"], c="magenta", s=0.5, alpha=0.1)
# %%
import pandas as pd

df_cb = pd.DataFrame(cb).T
# %%
