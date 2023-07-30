import matplotlib.pyplot as plt
import seaborn as sns

from imagecodecs import jpegxl_decode, jpegxl_encode, jpegxr_decode, jpegxr_encode

fig, axs = plt.subplots(ncols=4, figsize=(20, 5), dpi=200)

axs[0].imshow(imgs[3][17, 1600:1800, 400:600], interpolation="none")
axs[0].axis("off")
# title is the size of the object
axs[0].set_title(f"Original {ex.size * 2 / 1024 / 1024:.2f} MB")
for level, ax in zip([100, 99, 98], axs[1:]):
    res = jpegxl_decode(b := jpegxl_encode(imgs[3], level=level))[
        17, 1600:1800, 400:600
    ]
    ax.imshow(res, interpolation="none")
    ax.set_title(f"JPEG-XL, {level=}, {len(b)/ 1024 / 1024:.2f} MB")
    ax.axis("off")
