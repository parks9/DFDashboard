from __future__ import division


import os
import numpy as np
from scipy import ndimage
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.stats import sigma_clip
from ..cameras import camera_info as cam_info
from ..modeling import fit_pixels_1d_poly
from ..database import DFDatabaseManager
from ..flags import DarkFlags
from ..utils import dfio as io
from .. import utils
from .. import DFStruct


__all__ = ['check_stats',
           'check_column_fits',
           'check_zero_pixel_fraction',
           'create_master_dark']


def check_stats(path_or_pixels, bias_expected=1000, rms_expected=30,
                low_factor=0.5, high_factor=2, max_value_range=100):
    """
    Check image statistics of dark frame.

    Parameters
    ----------
    path_or_pixels : str or ndarray
        Dark image pixels of file name
    bias_expected : float (optional)
        Expected bias of the detector.
    rms_expected : flaot (optional)
        Expected rms of the dark.
    low_factor : float (optional)
        Factor below the expected values that is acceptable.
    hight_factor : float (optional)
        Factor above the expected values that is acceptable.
    max_value_range : float (optional)
        Maximum range in pixel values after taking median along columns 
        and smoothing with a 3 pixel median filter.

    Returns
    -------
    DFStruct object with attributes stats (the image stats) and
    flags (a list of flags, which will be empty if the image is good).
    """
    flags = DarkFlags()
    pixels = utils.load_path_or_pixels(path_or_pixels)
    value_range = ndimage.median_filter(np.nanmedian(pixels, axis=0), 3).ptp()
    stats = utils.measure.pixel_stats(pixels)

    if stats.median < 0:
        flags.set('negative_median')
    elif stats.median < low_factor * bias_expected:
        flags.set('low_median')
    elif stats.median > high_factor * bias_expected:
        flags.set('high_median')

    if stats.mad_rms < low_factor * rms_expected:
        flags.set('low_rms')
    elif stats.mad_rms > high_factor * rms_expected:
        flags.set('high_rms')

    if value_range > max_value_range:
        flags.set('large_value_range')

    return DFStruct(stats=stats, flags=flags, value_range=value_range)


def check_column_fits(path_or_pixels, order=5, stats=None, bias_expected=1000,
                      max_ratio_scale=1.015, max_model_var=1.1):
    """
    Fit Legendre polynomials to each column of dark and
    calculate quality metrics. The fits are carried out

    Parameters
    ----------
    path_or_pixels: str or ndarray
        Dark image pixels of file name
    order : int (optional)
        Order of polynomials.
    stats : DFStruct (optional)
        Image statistics -- the output from utils.calc_image_stats.
    bias_expected : float (optional)
        Expected bias of the detector.
    max_ratio_scale : float (optional)
        Maximum acceptable ratio of max and min of the dark image over the
        model: max(dark / model) / min(dark / model).
    max_model_var : float (optional)
        Maximum acceptable variation of the model quantified with the ratio:
        max(model) / min(model).

    Returns
    -------
    DFStruct object that contains flags (if any), the model, and
    the block reduced dark frame.
    """
    flags = DarkFlags()
    dark_pixels = utils.load_path_or_pixels(path_or_pixels)

    if stats is None:
        stats = utils.measure.pixel_stats(dark_pixels)

    # reset bias to bias_expected
    new_dark = dark_pixels + (bias_expected - stats.median)

    # median filter hot pixels and bad columns
    new_dark = ndimage.filters.median_filter(new_dark, size=(3, 3))

    # decrease the image size by a factor of 4
    new_dark = block_reduce(new_dark, 4, np.mean)

    # median filter again for cosmic rays
    new_dark = ndimage.filters.median_filter(new_dark, size=(5, 5))

    model = fit_pixels_1d_poly(new_dark, order=order, which='columns')

    # median filter model in x-direction to homogenize it
    model = ndimage.filters.median_filter(model, size=(1, 9))

    # divide image by median-filtered model
    ratio = block_reduce(new_dark / model , 10, np.mean)

    # measure the minimum and maximum in the block averaged ratio
    ratio_scale = np.max(ratio) / np.min(ratio[ratio != 0])

    if ratio_scale > max_ratio_scale:
        flags.set('HIGH_DARK_OVER_MODEL')

    mask = (model != 0.) & np.isfinite(model)
    model_var = np.max(model) / np.min(model[mask])

    if model_var > max_model_var:
        flags.set('HIGH_MODEL_VARIATION')

    results = DFStruct(flags=flags, model=model, dark_block_ave=new_dark)

    return results


def check_zero_pixel_fraction(path_or_pixels, max_zero_frac=0.1):
    """
    Check if large fraction of pixels are zero.

    Parameters
    ----------
    path_or_pixels: str or ndarray
        Dark image pixels of file name.
    max_zero_frac: float
        Maximum fraction of pixels with a value of zero.

    Returns
    -------
    DFStruct object that contains flags (if any) and
    the fraction of pixels with a value of zero.
    """
    flags = DarkFlags()
    frac_zero = utils.zero_pixel_fraction(path_or_pixels)
    if frac_zero > max_zero_frac:
        flags.set('ZERO_PIXEL_FRACTION')
    results = DFStruct(flags=flags, frac_zero=frac_zero)
    return results


def create_master_dark(image_path_list, save_path=None, 
                       sigma_lower=2, sigma_upper=3, stdfunc=np.ma.std, 
                       cenfunc=np.ma.median, run_label=None, 
                       serialno=None, rm_overscan=False, survey='NB', **kwargs):
    """
    Create master dark and optionally save fits image.

    Parameters
    ----------
    image_path_list : list-like
        Dark frame file names to be turned into master dark.
    save_path : str (optional)
        If not None, will save master dark in this directory.
    sigma_lower : float (optional)
        The number of standard deviations to use as the lower
        bound for the clipping limit.
    sigma_upper : float (optional)
        The number of standard deviations to use as the upper
        bound for the clipping limit.
    stdfunc : numpy.ma function (optional)
        The statistic or callable function/object used to compute
        the standard deviation about the center value.
    cenfunc : numpy.ma function (optional)
        The statistic or callable function/object used to compute
        the center value for the clipping.
    run_label : str (optional)
        It not None, will be appended to master dark file name to
        identify as a unique run.

    Returns
    -------
    master_dark : DFStruct
        Master dark pixels and header. 
    """
    bounds = cam_info[survey]['light_pixels'] if rm_overscan else None

    image_path_list = np.asarray(image_path_list)
    image_pixels = np.array([io.load_path_or_pixels(f,bounds=bounds) \
                             for f in image_path_list])

    image_pixels = sigma_clip(
        image_pixels, sigma_lower=sigma_lower, sigma_upper=sigma_upper,
        stdfunc=stdfunc, cenfunc=cenfunc, axis=0, **kwargs)
    pixels = np.float32(np.ma.average(image_pixels, axis=0).data)

    header = fits.getheader(image_path_list[0])
    header['NIMAGES'] = len(image_path_list)
    header['IMAGETYP'] = 'master dark'
    header.add_history(utils.default_header_history())

    if save_path is not None:
        utils.mkdir_if_needed(save_path)
        run_label = '' if run_label is None else '_' + run_label
        if (serialno is None) and ('SERIALNO' in header.keys()):
            serialno = header['SERIALNO']
        sn_label = '' if serialno is None else '_' + serialno
        exptime = header['EXPTIME']
        out_fn = f'master{sn_label}_dark_{exptime}{run_label}.fits'
        out_fn = os.path.join(save_path, out_fn)
        utils.write_pixels(out_fn, pixels, header)

    master_dark = DFStruct(pixels=pixels, header=header)

    return master_dark
