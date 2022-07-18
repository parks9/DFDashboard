import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage 
from astropy.io import fits
from astropy import units as u
from astropy.stats import sigma_clip

from ..detection import sextractor_object_mask
from ..cameras import camera_info as cam_info
from ..flags import FlatFlags
from ..improc import replace_dead_pixels
from .. import DFStruct
from .. import utils, logger


__all__ = [
    'create_master_flat',
    'count_negative_pixels', 
    'check_median_count_levels',
    'check_moon_proximity',
    'measure_ramp'
]


def create_master_flat(frame_id_list, db_hub, min_num_flats=7,
                       combinefunc=np.ma.median, apply_mask=True, 
                       mask_path=None, apply_sigma_clip=False, sigma_lower=2, 
                       sigma_upper=2, cenfunc=np.ma.median, stdfunc=np.ma.std, 
                       save_path=None, run_label=None, tilt=None, 
                       survey='NB', rm_overscan=False, **kwargs):
    """ 
    Create a master flat from a list of flat rame ID numbers. 

    Parameters
    ----------
    frame_id_list : list-like
        Frame id numbers (indexes) of flats that will go into master flat. 
    db_hub : DFDatabaseHub
        The database hub with classified flat info flats in the frame_id list.
    min_num_flats : int (optional)
        Require at least this many individual flats to make a master flat.
    combinefunc : numpy.ma function
        The callable function used to combine images for the master flat.
    apply_mask : bool (optional)
        If True, generate object masks and apply them during coaddition. 
    mask_path : str (optional)
        Save object masks here. Will use saved mask if it exists instead of 
        generating a new one to save time. 
    apply_sigma_clip : bool
        If True, sigma clip the images before combining.
    sigma_lower : float
        The number of standard deviations to use as the lower
        bound for the clipping limit.
    sigma_upper : float
        The number of standard deviations to use as the upper
        bound for the clipping limit.
    cenfunc : numpy.ma function
        The statistic or callable function/object used to compute
        the center value for the clipping.
    stdfunc : numpy.ma function (optional)
        The statistic or callable function/object used to compute
        the standard deviation about the center value.
    save_path : str (optional)
        If not None, will save master flat in this directory.
    run_label : str (optional)
        It not None, will be appended to master flat file name to
        identify as a unique run.
    tilt : float (optional)
        It not None, will be appended to master flat file name to
        identify the filter tilt of the flat images and saved into the header.
    """
    frames_db = db_hub.frames
    num_flats = len(frame_id_list)
    frame = frames_db.query('serialno,date', index=frame_id_list[0])
    sn, date = frame.serialno, frame.date
    if num_flats < min_num_flats:
        msg = '{} < {} is not enough flats to make a master flat for {} on {}'
        msg = msg.format(num_flats, min_num_flats, sn, date)
        logger.critical(msg)
        return None

    msg = 'Creating master flat for {} on {} using {} flats'
    logger.info(msg.format(sn, date, num_flats))

    bounds = cam_info[survey]['light_pixels'] if rm_overscan else None
    if bounds is not None:
        masked_pixel_count = np.zeros((bounds[1]-bounds[0],bounds[3]-bounds[2]))
    else:
        masked_pixel_count = np.zeros(frames_db.get_frame_shape(frame_id_list[0]))
    master_flat_pixels = []
    for frame_id in frame_id_list:
        flat = db_hub.get_dark_subtracted(frame_id)
        flat_pixels = flat.pixels.astype(float)
        if apply_mask:
            msg = 'Creating mask for: frame {}'.format(frame_id)
            if mask_path is None:
                logger.debug(msg)
                # TODO: add to config
                mask = sextractor_object_mask(
                    flat_pixels, back_size=32, detect_minarea=10, 
                    detect_thresh=5, dilate_npix=5)
            else:
                frame_fn = frames_db.get_frame_path(frame_id, True)
                mask_fn = frame_fn.replace('.fits', '_mask.fits')
                mask_fn = os.path.join(mask_path, mask_fn)
                if os.path.isfile(mask_fn):
                    logger.debug('Reading mask for: frame {}'.format(frame_id))
                    mask = utils.load_path_or_pixels(mask_fn)
                else:
                    logger.debug(msg)
                    mask = sextractor_object_mask(
                        flat_pixels, back_size=32, detect_minarea=10, 
                        detect_thresh=5, dilate_npix=5)
                    logger.debug('Writing mask for: frame {}'.format(frame_id))
                    utils.write_pixels(mask_fn, mask, flat.header) 
            flat_pixels[mask > 0] = np.nan
            masked_pixel_count += (mask > 0).astype(float)
        master_flat_pixels.append(flat_pixels/ np.nanmedian(flat_pixels))
    master_flat_pixels = np.ma.array(master_flat_pixels, 
                                     mask=np.isnan(master_flat_pixels))

    if apply_sigma_clip:
        master_flat_pixels = sigma_clip(
            master_flat_pixels, sigma_lower=sigma_lower, 
            sigma_upper=sigma_upper, stdfunc=stdfunc, cenfunc=cenfunc, axis=0)

    master_flat_pixels = combinefunc(master_flat_pixels, axis=0).data
    masked_frac = masked_pixel_count / num_flats 

    frac_heavily_masked_pixels = (masked_frac >= 0.5).sum() / masked_frac.size
    frac_fully_masked_pixels = (masked_frac == 1.0).sum() / masked_frac.size

    if apply_mask:
        msg = 'Fraction of {} masked pixels: {:.3f}%'
        logger.debug(msg.format('heavily', 100 * frac_heavily_masked_pixels))
        logger.debug(msg.format('fully', 100 * frac_fully_masked_pixels))

    dead_pix_kw = kwargs.pop('replace_dead_pixels', dict(padding=1))
    mflat_rdpix = replace_dead_pixels(master_flat_pixels, **dead_pix_kw)

    header = flat.header
    header['NIMAGES'] = num_flats
    header['IMAGETYP'] = 'master flat'
    header['COMBTYP'] = combinefunc.__name__
    header['NUMDEAD'] = mflat_rdpix.num_dead_pixels
    header['FMASK100'] = frac_fully_masked_pixels
    header['FMASK50'] = frac_heavily_masked_pixels
    header['MASK'] = str(apply_mask)
    if apply_sigma_clip:
        header['SIGLO'] = sigma_lower
        header['SIGHI'] = sigma_upper
    if tilt is not None:
        header['TILT'] = tilt
    header.add_history(utils.default_header_history())

    master_flat = DFStruct(header=header,
                           pixels=mflat_rdpix.pixels.astype(np.float32), 
                           masked_frac=masked_frac)
    
    if save_path is not None:
        utils.mkdir_if_needed(save_path)
        fn = db_hub.mcals.build_flat_path(date, sn)
        if run_label is not None:
            fn = fn.replace('.fits', f'_{run_label}.fits')
        if tilt is not None:
            fn = fn.replace('.fits', f'_{tilt}.fits')
        utils.write_pixels(fn, master_flat.pixels, master_flat.header)

    return master_flat


def count_negative_pixels(image_path, max_negative_counts=1000):
    """ 
    Return flag if the number of pixels with negative counts
    exceeds some threshold.

    Parameters
    ----------
    image_path : str
        Full path to the dark-subtracted flatfield image.
    max_negative_counts : int
        Flag image as bad if the number of negative pixels
        crosses this threshold.

    Returns
    ------
    flag : str
        String describing the results
    """
    flags = FlatFlags()
    image_pixels = utils.load_path_or_pixels(image_path)
    num_negatives = np.sum(image_pixels < 0)

    if num_negatives > max_negative_counts:
        flags.set('negative_pixels')

    return flags


def check_median_count_levels(image_path, min_expected=5000., 
                              max_expected=40000.):
    """ 
    Return flag if image has anomalously high or low (median) counts.

    Parameters
    ----------
    image_path : str
        Full path to the dark-subtracted flatfield image.
    min_expected : float
        Dark-subtracted flats should have median counts above this threshold.
    max_expected : float
        Dark-subtracte flats should have median counts below this threshold.

    Returns
    ------
    flag : str
        String describing the results

    NOTES
    -----
    High counts indicates saturation.
    Low counts indicates a lens might be stopped down
    (or flats were mistakenly taken with closed domes, etc.)
    """
    flags = FlatFlags()

    image_pixels = utils.load_path_or_pixels(image_path)
    median_counts = np.median(image_pixels)

    if median_counts < min_expected:
        flags.set('low_median_counts')
    elif median_counts > max_expected:
        flags.set('high_median_counts')

    return flags


def check_moon_proximity(path_or_header, altaz_tolerance=2.0,
                         max_moon_alt=-3, min_targetmoon_sep=45):
    """ 
    Check whether the moon is (a) too high or (b) too close to the target
    at the time of observations.

    Parameters
    ----------
    path_or_header: str or astropy.fits.Header
        Image path or fits header.
    max_moon_alt : float
        Flag if observations were acquired when the moon was higher than
        this altitude [unit:deg].
    min_targetmoon_sep : float
        Flag if observations were acquired when the target-moon separation
        was less than this angular distance [unit:deg].
    altaz_tolerance : float
        Flag if utils.calc_moon_data returns incorrect moon data.
        The check: compare the differnece in altitude and azimuth between
        the image header and the ephem outputs. [unit:deg]

    Returns
    ------
    flag : str
        String describing the results

    NOTES
    -----
    Sometimes the headers have alt/az coordinates that do *not* match the
    radec. This seems to happpen at fixed frame number, across cameras. By
    comparison with the images that have correct/consistent information, the
    issue seems to be with alt/az rather than radec. Therefore, we will write
    out a flag for book-keeping, but this should *not* be grounds for
    rejecting a flat.
    """
    flags = FlatFlags()

    hdr = utils.load_path_or_header(path_or_header)
    datetime = hdr['DATE']
    alt = hdr['ALTITUDE']
    az = hdr['AZIMUTH']

    skycoord = utils.to_skycoord([hdr['OBJCTRA'], hdr['OBJCTDEC']])
    moon = utils.get_moon_status(skycoord, datetime)
    if moon.success:
        altdiff = np.abs(moon.target_alt.deg - alt)
        azdiff = np.abs(moon.target_az.deg - az)
        if altdiff > altaz_tolerance or azdiff > altaz_tolerance:
            flags.set('bad_header_altaz')
        if moon.alt.deg > max_moon_alt:
            flags.set('moon_up')
        if moon.sep_from_target.deg < min_targetmoon_sep:
            flags.set('moon_near')
    else:
        flags.set('no_moon_data')

    return flags 


def check_streakiness(image_path):
    """ check for tracking errors

    Parameters
    ----------
    image_path : str
        Full path to the flatfield frame of interest.

    Returns
    -------
    flag : str
        String describing the results
    """
    pass


def measure_ramp(path_or_pixels, max_allowed_slope=5e-6, **kwargs):
    """ Measure the slope across an image, flag if too steep.

    Parameters
    ----------
    image_path : str
        Full path to the image
    max_allowed_slope : float
        Flag if the linear coefficient (slope) exceeds this threshold

    Return
    ------
    flag : str
        String describing the results
    """
    flags = FlatFlags()

    linear_kw = kwargs.pop('measure_linear_ramp', {})
    slope = utils.linear_ramp(path_or_pixels, **linear_kw)

    if slope >= np.float(max_allowed_slope):
        flags.set('bad_slope')

    results = DFStruct(slope=slope, flags=flags)

    return results
