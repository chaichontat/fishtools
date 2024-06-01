# %%
from pathlib import Path

import numpy as np
import tifffile
from astropy import modeling

# fitter = modeling.fitting.LevMarLSQFitter()
# model = modeling.models.Gaussian1D()  # depending on the data you need to give some initial values
# fitted_model = fitter(model, range(psf.shape[1]), psf[17, 18])
# print(fitted_model)

# %%
psfs = [Path(f"data/psf_{i}_selected.tif") for i in [405, 488, 560, 650]]
mins = []
for p in psfs:
    curr = np.inf
    psf = tifffile.imread(p)
    for i in range(12, 25):
        fitter = modeling.fitting.LevMarLSQFitter()
        model = modeling.models.Gaussian2D()
        fitted_model = fitter(model, *np.meshgrid(range(psf.shape[1]), range(psf.shape[2])), psf[i])
        curr = min(curr, np.mean([fitted_model.y_stddev.value, fitted_model.x_stddev.value]))
    mins.append(curr)
mins
# σ ~ 1.65 for all channels. FWHM = 2.355 * σ ~ 3.9 -> 400 nm
