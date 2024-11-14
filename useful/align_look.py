# %%
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from starfish.types import Axes, Features, Levels

sns.set_theme()

decoded_spots, spot_intensities = pickle.loads(
    Path("/mnt/archive/starmap/e155/e155_working/analysis/deconv/registered--down/reg-0360.pkl").read_bytes()
)

codebook = json.loads(Path("/fast2/fishtools/starwork3/ordered/genestar.json").read_text())
# codebook, used_bits, names, arr_zeroblank = load_codebook(
#     Path("/fast2/fishtools/starwork3/ordered/genestar.json"), exclude={"Malat1-201"}
# )

rand = np.random.default_rng(0)

# %%
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(
    decoded_spots.coords["radius"] + rand.normal(0, 0.05, size=decoded_spots.shape[0]),
    decoded_spots.coords["distance"],
    # c=decoded_spots.coords["passes_thresholds"],
    # c=np.linalg.norm(decoded_spots, axis=2),
    # c=np.where(
    #     is_blank[list(map(names_l.get, initial_spot_intensities.coords["target"].to_index().values))], 1, 0
    # ).flatten(),
    alpha=0.2,
    cmap="bwr_r",
    s=2,
)
ax.set_xlabel("Radius")
ax.set_ylabel("Distance")
plt.legend()
# %%

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(
    decoded_spots.coords["radius"] + rand.normal(0, 0.05, size=decoded_spots.shape[0]),
    decoded_spots.coords["distance"],
    c=~decoded_spots["target"].str.startswith("Blank"),
    # c=np.linalg.norm(decoded_spots, axis=2),
    # c=np.where(
    #     is_blank[list(map(names_l.get, initial_spot_intensities.coords["target"].to_index().values))], 1, 0
    # ).flatten(),
    alpha=0.2,
    cmap="bwr",
    s=2,
)
ax.set_xlabel("Radius")
ax.set_ylabel("Distance")

# %%
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(
    np.linalg.norm(np.array(decoded_spots), axis=(1, 2)),
    decoded_spots.coords["radius"] + rand.normal(0, 0.05, size=decoded_spots.shape[0]),
    c=~decoded_spots["target"].str.startswith("Blank"),
    # c=np.linalg.norm(decoded_spots, axis=2),
    # c=np.where(
    #     is_blank[list(map(names_l.get, initial_spot_intensities.coords["target"].to_index().values))], 1, 0
    # ).flatten(),
    alpha=0.2,
    cmap="bwr",
    s=2,
)
# set semilogx
plt.xscale("log")
plt.legend()


# %%
spot_intensities = decoded_spots.loc[decoded_spots["radius"] > 1.6]
shits = spot_intensities.where(spot_intensities["target"].str.startswith("Blank"), drop=True)
# %%
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.scatter(
    decoded_spots.coords["radius"],
    np.linalg.norm(np.array(decoded_spots), axis=(1, 2)),
    c=~decoded_spots["target"].str.startswith("Blank"),
    # c=np.linalg.norm(decoded_spots, axis=2),
    # c=np.where(
    #     is_blank[list(map(names_l.get, initial_spot_intensities.coords["target"].to_index().values))], 1, 0
    # ).flatten(),
    alpha=0.2,
    cmap="bwr",
    s=2,
)
# set log
ax.set_xscale("log")
ax.set_yscale("log")


# %%
plt.scatter(
    shits.coords["radius"],
    np.linalg.norm(np.array(shits), axis=(1, 2)),
    # c=np.linalg.norm(decoded_spots, axis=2),
    # c=np.where(
    #     is_blank[list(map(names_l.get, initial_spot_intensities.coords["target"].to_index().values))], 1, 0
    # ).flatten(),
    alpha=0.2,
    cmap="bwr",
    s=2,
)

ok = spot_intensities.where(spot_intensities > 0.45 - 0.3 * spot_intensities.coords["radius"], drop=True)
print(len(ok))
# %%

plt.hist(spot_intensities.coords["radius"], bins=100)
plt.hist(
    spot_intensities.where(decoded_spots["target"].str.startswith("Blank"), drop=True).coords["distance"],
    bins=100,
)
plt.yscale("log")

# %%
norm = np.linalg.norm(np.array(spot_intensities), axis=(1, 2))

plt.hist(norm, bins=100)
plt.hist(np.linalg.norm(np.array(shits), axis=(1, 2)), bins=100)
plt.yscale("log")


# %%

using = ok


dist = using[Features.AXIS].to_dataframe().groupby("target").agg(dict(distance="mean"))

dist["blank"] = dist.index.str.startswith("Blank")
genes, counts = np.unique(
    using[Features.AXIS][Features.TARGET],
    return_counts=True,
)
gc = dict(zip(genes, counts))
percent = sum([v for k, v in gc.items() if k.startswith("Blank")]) / counts.sum()
c = pd.DataFrame.from_dict(gc, orient="index").sort_values(0)
# c = c[c[0] > 1]
c["color"] = c.index.str.startswith("Blank")
c["color"] = c["color"].map({True: "red", False: "blue"})

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.bar(c.index, c[0], color=c["color"], width=1, align="edge", linewidth=0)
ax.set_xticks([])
ax.set_yscale("log")
ax.set_xlabel("Gene")
ax.set_ylabel("Count")
plt.tight_layout()


# %%
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
baddies = c[c["color"] == "red"]
ax.bar(c.index[-20:], c[0][-20:], color=c["color"][-20:], width=1, align="edge", linewidth=0)
ax.set_yscale("log")
ax.set_xticklabels(c.index[-20:], rotation=45, ha="right")
# %%


# %%


def plot_decoded(spots):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.scatter(spots.coords["x"], -spots.coords["y"], s=0.1, c="red", alpha=0.5)
    # plt.imshow(prop_results.label_image[0])
    ax.set_title("PixelSpotDecoder Labeled Image")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()


plot_decoded(spot_intensities)
# %%


# %%


def plot_bits(spots):
    reference = np.zeros((len(spots), spots.shape[-1] + 4))
    for i, arr in enumerate(list(map(codebook.get, spots.target.values))):
        for a in arr:
            reference[i, a - 1] = 1

    fig, axs = plt.subplots(figsize=(2, 8), ncols=1, dpi=200, facecolor="white")
    axs = [axs]
    axs[0].imshow(spots.squeeze(), vmax=np.percentile(spots.squeeze(), 90), zorder=1)
    # axs[0].axis("off")
    # axs[1].imshow(reference)
    # axs[1].axis("off")
    # for ax in axs:
    #     ax.set_aspect("equal")
    axs[0].set_title("Neurod6-201")
    axs[0].set_xlabel("Bit")
    axs[0].set_ylabel("Spot")

    # Set axis to be white

    plt.tight_layout()

    return fig, axs


plot_bits(ok.where(ok.target == "Neurod6-201", drop=True))

# %%
df = pd.DataFrame(
    {
        "target": spot_intensities.coords["target"],
        "x": spot_intensities.coords["x"],
        "y": spot_intensities.coords["y"],
    }
)


sns.scatterplot(data=df[df["target"] == "Neurod1-201"], x="x", y="y", hue="target", s=10, legend=True)

# %%
# Example of how to access the spot attributes
print(f"The area of the first spot is {prop_results.region_properties[0].area}")

# View labeled image after connected componenet analysis
# View decoded spots overlaid on max intensity projected image
single_plane_max = imgs.reduce({Axes.ROUND, Axes.CH, Axes.ZPLANE}, func="max")
fig, axs = plt.subplots(ncols=2, figsize=(10, 5), dpi=200)
axs[1].imshow(prop_results.label_image[0], vmax=1)
axs[1].set_title("Decoded", loc="left")


axs[0].imshow(single_plane_max.xarray[0].squeeze(), cmap="gray")
axs[0].set_title("Raw", loc="left")
for ax in axs:
    ax.axis("off")
fig.tight_layout()
