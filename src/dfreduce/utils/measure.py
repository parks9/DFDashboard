"""
Functions for measuring things in/from images.
"""
import numpy as np
from astropy import stats
from .dfio import load_path_or_pixels
from .. import DFStruct, utils


__all__ = ['linear_ramp', 'pixel_stats', 'zero_pixel_fraction']


def linear_ramp(path_or_pixels, row_slice='center_row', clip_pixels=1.1):
    """ 
    Measure the slope of a 'ramp' in an image

    Parameters
    ----------
    path_or_pixels : str or numpy array
        Full path to image or its pixels. 
    row_slice : str or numpy slice object (optional)
        Default value is 'center_row'. To measure the slope of a different
        row in the image, create a slice with np.s_.
    clip_pixels : float
        Clip pixels that exceed this value.

    Returns
    -------
    The absolute value of the coefficient of the linear fit.
    """
    pixels = load_path_or_pixels(path_or_pixels)

    x_pixels = np.arange(pixels.shape[1])
    if row_slice == 'center_row':
        s_ = np.s_[int(pixels.shape[0]/2): int(pixels.shape[0]/2) + 1, :]
        y_pixels = pixels[s_]
    else:
        y_pixels = pixels[pixels_slice]

    y_pixels = np.squeeze(y_pixels)
    y_pixels /= np.ma.mean(y_pixels)

    x = x_pixels[y_pixels < clip_pixels]
    y = y_pixels[y_pixels < clip_pixels]

    abs_coeff = np.abs(np.polyfit(x, y, 1)[0])

    return abs_coeff


def pixel_stats(path_or_pixels, pixels_slice=np.s_[1066:1466, 1476:1876]):
    """
    Calculate image statistics: mean, rms, median, mad, and mad_rms.

    Parameters
    ----------
    path_or_pixels: str or ndarray
        The image as a numpy array or the file name.
    pixels_slice : numpy slice object (optional)
        Create with np.s_. If None, will use full image.

    Returns
    -------
    results : dfreduce.DFStruct
        The image statistics.
    """
    pixels = load_path_or_pixels(path_or_pixels)
    if pixels_slice is None:
        sub_pixels = pixels
    else:
        sub_pixels = pixels[pixels_slice]
    mean = np.mean(sub_pixels)
    stddev = np.std(sub_pixels)
    median = np.median(sub_pixels)
    mad = stats.median_absolute_deviation(sub_pixels)
    mad_rms = 1.48 * mad
    results = DFStruct(mean=mean, stddev=stddev,
                       median=median, mad=mad, mad_rms=mad_rms)
    return results


def zero_pixel_fraction(path_or_pixels):
    pixels = utils.load_path_or_pixels(path_or_pixels)
    frac_zero = (pixels == 0).sum() / pixels.size
    return frac_zero
