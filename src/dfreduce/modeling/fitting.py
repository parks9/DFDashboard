import itertools
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy import optimize
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from .models import CosineFlatField
from ..utils import load_path_or_pixels
from .. import DFStruct, logger


__all__ = [
    'fit_2d_poly',
    'fit_pixels_1d_poly',
    'fit_flat_field_1d',
]


def fit_2d_poly(z, deg, include_cross_terms=True):
    if type(deg) == int:
        deg = np.array([deg, deg])
    else:
        deg = np.asarray(deg)
    nrows, ncols = z.shape
    yy, xx = np.mgrid[:nrows, :ncols]
    # TODO: scale these to be between 0 - 1 to avoid 
    # machine precision issues. 
    x = xx.flatten() 
    y = yy.flatten()
    z = z.flatten() 
    vander = poly.polyvander2d(x, y, deg)
    if not include_cross_terms:
        ij = itertools.product(range(deg[0] + 1), range(deg[1] + 1))
        ij = np.array(list(ij))
        mask = (ij[:, 0] != 0) & (ij[:, 1] != 0)
        vander[:, mask] = 0
    coeff = np.linalg.lstsq(vander, z, rcond=-1)[0]
    coeff = coeff.reshape(deg + 1)
    poly_fit = poly.polyval2d(xx, yy, coeff)
    results = DFStruct(coeff=coeff, pixels=poly_fit)
    return results


def fit_pixels_1d_poly(image, order=5, sigma=3, maxiters=5, which='columns'):
    """
    Fit Legendre polynomials to sigma-clipped rows or columns of image.

    Parameters
    ----------
    image : ndarray or str
        The image pixels or file name.
    order : int (optional)
        The order of the polynomials.
    sigma : float (optional)
        Number of sigma for sigma clipping.
    maxiters : int (optional)
        Maximum number of iterations for the sigma clipping.
    which : str
        Which 1d slice: rows or columns

    Returns
    -------
    model : ndarray
        The best fit model.
    """
    pixels = load_path_or_pixels(image)

    # create output model image and astropy fitter
    model = np.zeros(pixels.shape)
    fitter = fitting.LinearLSQFitter()

    iter_i = dict(row=0, rows=0, col=1, cols=1, columns=1)[which]
    fit_i = 1 if iter_i == 0 else 0
    slicer = (lambda i: np.s_[i, :]) if fit_i == 1 else (lambda i: np.s_[:, i])

    for line in range(pixels.shape[iter_i]):

        # sigma clip column before fitting
        s_ = slicer(line)
        col_sig_clipped = sigma_clip(pixels[s_], sigma, maxiters=maxiters)
        data = col_sig_clipped.data
        good = ~col_sig_clipped.mask

        # use the median of each column as the
        # initial guess for c0, otherwise normalization is way off
        c0 = np.median(data[good])
        leg_poly = models.Legendre1D(degree=order, c0=c0)

        # perform fit using weight = 0 for outlier pixels
        x = np.arange(pixels.shape[fit_i])
        f = fitter(leg_poly, x, data, weights=good.astype(int))

        model[s_] = f(x)

    return model


def fit_flat_field_1d(x, y, y_err, p0, fit_func='cos_model', bounds=None, **kwargs):
    func_dict = dict(cos_model=CosineFlatField)
    fitter = func_dict[fit_func]
    method = kwargs.pop('method', 'L-BFGS-B')
    flatmodel = fitter(x, y, y_err)
    results = optimize.minimize(
        lambda p: -flatmodel(p), p0, method=method,
        bounds=bounds, **kwargs)
    if results.success:
        r = y - flatmodel.model(x, results.x)
        nu = len(y) - len(p0)
        chi2_dof = np.sum((r / y_err)**2) / nu
        results = DFStruct(p=results.x,
                           success=results.success,
                           chi2_dof=chi2_dof)
    else:
        logger.warning('1D flat-field fitting failed with message:')
        logger.warning(results.message)
        results = DFStruct(p=None, success=False, chi2_dof=99999)
    return results
