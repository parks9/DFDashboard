"""
Core functions for processing and/or manipulating images.
"""
import numpy as np
from scipy import ndimage
from astropy.wcs import WCS
from astropy.io import fits
from ..utils import dfio as io
from ..astrometry import pixel_area_map
from .. import DFStruct
from ..cameras import camera_info as cam_info
from .. import logger


try:
    from astrometry.util.util import Tan, Sip
    from astrometry.util.resample import resample_with_wcs, OverlapError
    DUSTIN_MAGIC = True
except:
    logger.warning('The magic of Dustin not found -- no resampling')
    DUSTIN_MAGIC = False

if DUSTIN_MAGIC:
    try:
        import fitsio
    except:
        logger.warning('fitsio not found -- no resampling')
        DUSTIN_MAGIC = False


__all__ = ['crop_image',
           'divide_by_master_flat', 
           'grid_image',
           'replace_dead_pixels', 
           'resample_image',
           'subtract_master_dark']


def crop_image(path_or_pixels, new_shape, save_fn=None, save_header=None):
    image = io.load_path_or_pixels(path_or_pixels)

    o_r, o_c = image.shape # old_row, old_col
    n_r, n_c = new_shape   # new_row, new_col
    odd = (np.array(new_shape) % 2).astype(int)

    s_ = np.s_[
        int(o_r / 2.) - int(n_r / 2.): int(o_r / 2. ) + int(n_r / 2.) + odd[0],
        int(o_c / 2.) - int(n_c / 2.): int(o_c / 2. ) + int(n_c / 2.) + odd[1]
    ]

    cropped_image = image[s_]
    
    if save_fn is not None:
        if save_header is not None:
            msg = 'Image cropped from {} to {}'.format(image.shape, new_shape)
            kw = dict(msg=msg, NAXIS1=n_r, NAXIS2=n_c)
            save_header = io.update_header(save_header, **kw)
        io.write_pixels(save_fn, cropped_image, header=save_header)
    
    return cropped_image


def grid_image(shape, seg_per_row, seg_per_col=None):
    seg_per_col = seg_per_row if seg_per_col is None else seg_per_col
    n_seg = seg_per_row * seg_per_col
    labels = np.arange(1, n_seg + 1, dtype=int)
    
    row_seg = int(shape[0]) / int(seg_per_row)
    col_seg = int(shape[1]) / int(seg_per_col)
    
    grid = labels.reshape(seg_per_row, seg_per_col).\
                  repeat(row_seg, axis=0).\
                  repeat(col_seg, axis=1)
    
    if grid.shape == shape:
        seg_grid = grid
    else:
        s_init = grid.shape
        seg_grid = np.zeros(shape)
        seg_grid[:s_init[0], :s_init[1]] = grid
        
        if seg_grid.shape[0] > s_init[0]:
            seg_grid[s_init[0]:, :] = seg_grid[s_init[0] - 1, :].\
                                            reshape(1, shape[1]).\
                                            repeat(shape[0] - s_init[0], 0)
            
        if seg_grid.shape[1] > s_init[1]:
            seg_grid[:, s_init[1]:] = seg_grid[:, s_init[1] - 1].\
                                            reshape(shape[0], 1).\
                                            repeat(shape[1] - s_init[1], 1)
            
    results = DFStruct(pixels=seg_grid.astype(int),
                       labels=labels.astype(int),
                       slices=ndimage.find_objects(seg_grid.astype(int)))
    
    return results



def divide_by_master_flat(path_or_pixels, flat_path_or_pixels, 
                          path_or_header=None, save_fn=None, 
                          norm_func=np.ma.mean, clip_pixel_lim=None,
                          survey='NB', rm_overscan=False):
    bounds = cam_info[survey]['light_pixels'] if rm_overscan else None
    image = io.load_path_or_pixels(path_or_pixels, bounds=bounds)
    master_flat = io.load_path_or_pixels(flat_path_or_pixels, bounds=bounds)
    master_flat = master_flat / norm_func(master_flat)
    image_ff = image / master_flat
    if clip_pixel_lim is not None:
        image_ff = np.clip(image_ff, *clip_pixel_lim)
    kw = dict(msg='Image flat fielded', normfunc=norm_func.__name__)
    if path_or_header is not None:
        header = io.update_header(path_or_header, **kw)
    elif type(path_or_pixels) == str or type(path_or_pixels) == np.str_:
        header = io.update_header(path_or_pixels, **kw)
    else:
        header = None
    if save_fn is not None:
        io.write_pixels(save_fn, image_ff, header=header)
    results = DFStruct(pixels=image_ff, header=header)
    return results


def replace_dead_pixels(image_pixels, padding=1, dead_value=0):
    """ 
    Replace dead pixels in an image with the median of their surrounding
    pixel values. (Watch out for edges!)

    Parameters
    ----------
    image_pixels : ndarray
        Image as a numpy array.
    padding : int
        Sets the size of the region in which to compute the median value of
        "nearby" pixels.

    Returns
    -------
    image_pixels : ndarray
        The image, with dead pixels replaced.
    num_dead_pixels : int
        The number of dead pixels replaced.
    """
    padding = int(padding)

    dead_pixel_indices = np.argwhere(image_pixels==dead_value)
    num_dead_pixels = len(dead_pixel_indices)

    for ind in dead_pixel_indices:
        row, col = ind

        min_row = np.max([0, row - padding])
        max_row = np.min([row + padding + 1, image_pixels.shape[0]])
        min_col = np.max([0, col - padding])
        max_col = np.min([col+padding+1, image_pixels.shape[1]])

        med_value = np.median(image_pixels[min_row:max_row, min_col:max_col])
        image_pixels[row, col] = med_value

    results = DFStruct(pixels=image_pixels, num_dead_pixels=num_dead_pixels)

    return results


def resample_image(path_or_pixels, ra_c, dec_c, pixscale, width, height,
                   fitsio_header=None, apply_pam=True, normalize_at='peak',
                   interp_type='Lanczos'):
    """
    Resample image using Dustin's magic.

    Thanks Dustin <3

    Parameters
    ----------
    frame_path : str
        Fits file name.
    ra_c : float
        Central RA of resampled image in degrees.
    dec_c : float
        Central DEC of resampled image in degrees.
    pixscale : float
        The desired pixel scale.
    width : int
        The width of the image in pixels.
    height : int
        The height of the image in pixels.
    interp_type : str
        Interpolation method (lanczos or nearest)

    Return
    ------
    results : DFStruct
        results.resampled_image : ndarray
            The resampled image.
        results.wcs : astropy.wcs.WCS
            Astropy WCS object.
    """
    if not DUSTIN_MAGIC:
        raise Exception('You do not have the magic of Dustin.')

    header = fitsio_header
    if not (type(path_or_pixels) == str or type(path_or_pixels) == np.str_):
        msg = 'Must provide *fitsio* header if pixels are given!'
        assert header is not None, msg
        assert type(header) == fitsio.header.FITSHDR, msg
    elif header is None:
        header = fitsio.read_header(path_or_pixels)
    pixels = io.load_path_or_pixels(path_or_pixels)

    ps = pixscale / 3600.
    x_c = width / 2 + 0.5
    y_c = height / 2 + 0.5
    target_wcs = Tan(ra_c, dec_c, x_c, y_c, -ps, 0., 0.,
                     ps, float(width), float(height))
    resampled = np.zeros((height, width), np.float32)
    _wcs = Sip(header)

    # TODO: find a better way to handle different header types
    astropy_header = fits.Header()
    for k, v in dict(fitsio_header).items():
        astropy_header[k] = v
    astropy_header['EPOCH'] = 2000.0
    pam = pixel_area_map(astropy_header, normalize_at=normalize_at)

    if apply_pam:
        pixels = pixels.astype(np.float64)
        pixels = pixels / pam
    try:
        if interp_type.lower() == 'lanczos':
            Yo, Xo, Yi, Xi, (rim,) = resample_with_wcs(target_wcs,
                                                       _wcs, [pixels])
            resampled[Yo, Xo] += rim
        elif interp_type.lower() == 'nearest':
            Yo, Xo, Yi, Xi, rims = resample_with_wcs(target_wcs, _wcs)
            resampled[Yo, Xo] += pixels[Yi, Xi]
        else:
            raise Exception(interp_type + ' is not a valid interp type')
    except OverlapError:
        logger.critical('WCS frames do not overlap!')
        return False

    # create an astropy wcs object
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = target_wcs.crval
    w.wcs.crpix = target_wcs.crpix
    w.wcs.cdelt = target_wcs.cd[0], target_wcs.cd[-1]
    w.pixel_shape = width, height
    results = DFStruct(pixels=resampled, wcs=w, pam=pam)

    return results


def subtract_master_dark(path_or_pixels, dark_path_or_pixels, 
                         path_or_header=None, save_fn=None,
                         save_header=None, survey='NB', rm_overscan=False):
    """
    Subtract master dark from image.

    Parameters
    ----------
    path_or_pixels : str or ndarray
        Full path to the image fits file or the image as a numpy array.
    dark_path_or_pixels : str or ndarray
        Full path to the master dark or the image as a numpy array.
    save_fn : str (optional)
        Fits file name to save the dark-subtracted frame to.

    Returns
    -------
    image_ds : ndarray
        The dark-subtracted image.
    """
    bounds = cam_info[survey]['light_pixels'] if rm_overscan else None
    image = io.load_path_or_pixels(path_or_pixels,bounds=bounds)
    master_dark = io.load_path_or_pixels(dark_path_or_pixels,bounds=bounds)
    image_ds = image - master_dark
    kw = dict(msg='Image dark subtracted')
    if path_or_header is not None:
        header = io.update_header(path_or_header, **kw)
    elif type(path_or_pixels) == str or type(path_or_pixels) == np.str_:
        header = io.update_header(path_or_pixels, **kw)
    else:
        header = None
    if save_fn is not None:
        io.write_pixels(save_fn, image_ds, header=header)
    results = DFStruct(pixels=image_ds, header=header)
    return results
