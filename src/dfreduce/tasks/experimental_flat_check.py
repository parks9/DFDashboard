import os
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from .fitting import fit_flat_1d
from .log import logger
from .cameras import camera_info

_ff_default_init = dict(
    bias=1000.,
    lightlevel=64000.0, 
    xc=1690.0,
    A=0.1, 
    w1=1400, 
    B=0.002, 
    w2=650.0
)

_ff_default_bounds = [
    (500, 1500), 
    (60000, 68000),
    (700, 2000), 
    (0, 1), 
    (0.01, np.inf), 
    (0, 1), 
    (0.01, np.inf), 
]


__all__ = ['assess_flat_field']


def assess_flat_field(flat, init_params={}, 
                      bounds=_ff_default_bounds, survey='UW'):
    """
    Assess quality of flat field using Cosine model. 

    Paramters
    ---------
    flat : ndarray or str
        The flat field image array or file name.

    NOTE
    ---- 
    The flat is assumed to be dark subtracted
    """

    if type(flat) == str:
        logger.debug('Loading flat file ' + flat)
        flat = fits.getdata(flat)

    params = _ff_default_init.copy()
    for k, v in init_params.items():
        params[k] = v
    
    # make per-column threshold nsig above mean
    ydata = flat.copy()
    nsig = 15
    thresh =  np.mean(ydata, axis=0) + nsig * np.std(ydata, axis=0)

    # count number of pixels below threshold in each column
    num_pix = np.ones_like(ydata)
    ydata[ydata > thresh] = np.nan
    num_pix[np.isnan(ydata)] = np.nan
    num_pix = np.nansum(num_pix,  axis=0)

    # calculate sum and error of each column
    detector = camera_info[survey]
    y = np.nansum(ydata, axis=0)
    g = detector.gain.to('electron / adu').value
    rn = detector.readnoise.to('electron').value
    var_electron = np.nansum(g * ydata, axis=0)
    y_err = np.sqrt(var_electron + num_pix * rn**2) / g

    # fit 1D model to the projected flat field
    squeeze = 100 # the edges misbehave
    x = np.arange(0, len(y))
    scale = np.mean(y)
    x = x[squeeze: -squeeze]
    y = y[squeeze: -squeeze] / scale
    y_err = y_err[squeeze: -squeeze] / scale

    p_names = ['bias', 'lightlevel', 'xc', 'A', 'w1', 'B', 'w2']
    p0 = [params[v] for v in p_names]
    results = fit_flat_1d(x, y, y_err, p0, bounds=bounds)

    return results
