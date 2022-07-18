import numpy as np
from scipy import signal
from astropy.table import Table 
from astropy.wcs import WCS
from astropy import units as u

from ..flags import LightFlags
from ..modeling import fit_pixels_1d_poly, fit_2d_poly
from ..detection import sextractor, create_source_map, sextractor_sky_model
from ..detection import extract_bright_stars
from ..cameras import get_filter_name
from .. import utils, DFStruct, improc
from .. import logger


__all__ = ['calculate_zp',
           'check_double_stars', 
           'check_image_quality', 
           'check_pointing', 
           'embed_air_mass',
           'embed_zp_space',
           'find_halos',
           'model_sky', 
           'model_sky_photutils']


def calculate_zp(path_or_pixels, bandpass, catalogue='APASS', catalogue_dir=None, 
                sep_max=2.0, mag_min=14.0, mag_max=16.5, min_num_matched_obj=100,
                run_label='zp', **kwargs):
    """
    Calculate the zeropoint of an image using a reference catalogue.

    Parameters
    ----------
    path_or_pixels : str or ndarray
        Full image path or the image pixels as a numpy array.
    bandpass: str
        Name of the bandpass used in image, match with the reference catalogue
    catalogue : str or astropy Table
        Name of the reference catalogue to use as reference for determining zp
        or reference catalogue as an astropy Table. Default is APASS.
    catalogue_dir : directory (optional)
        The path to the directory where the reference catalogue files are stored.
        Must specify if using APASS catalogues.
    sep_max : float (optional)
        Maximum allowable star separation between image and
        reference catalogue (arcsec)
    mag_min : float (optional)
        Exclude sources brighter than this from analysis.
    mag_max : float (optional)
        Exclude sources fainter than this from analysis.
    min_num_matched_obj : int (optional)
        Minimum number of matched sources between the image catalogue and the
        reference catalogue. Default is 100.
    Returns
    -------
    results : DFStruct object
        Differences between the instrumental mags and the reference mags 
        (mag_diffs), the median, mean, and standard deviation of the image 
        zero point (zp_median, zp_mean, and zp_stddev), and the reference 
        catalog (ref_cat).
    """

    imagecat = extract_bright_stars(
        path_or_pixels,
        extra_params='X_WORLD,Y_WORLD,FLUXERR_AUTO',
        run_label=run_label, 
        **kwargs
    )
                              
    if isinstance(catalogue, Table):
        refcat = catalogue
    elif catalogue=='APASS':
        if catalogue_dir is None:
            logger.error('calculate_zp: Need to specify catalogue_dir.')
            return None
        refcat = utils.load_apass_in_region(catalogue_dir,cat=imagecat)
    else:
        logger.error('calculate_zp: Only APASS catalogues are currently supported.')
        return None

    bandpass = bandpass.lower()
    imagecat_match,refcat_match = utils.match_catalogues(imagecat,refcat,bandpass,
        sep_max=sep_max, mag_min=mag_min, mag_max=mag_max)

    if len(refcat_match) < min_num_matched_obj:
        imname = path_or_pixels if type(path_or_pixels)==str else 'input array'
        logger.error(f'calculate_zp: Only {len(refcat_match)} matched sources '+
            f'found--unable to flatten photometry across the field for {imname}')
        return None

    zps = refcat_match[bandpass] - imagecat_match[bandpass]

    results = DFStruct(mag_diffs=zps,
                       zp_median=np.median(zps),
                       zp_mean=np.mean(zps),
                       zp_stddev=np.std(zps), 
                       img_cat_match=imagecat_match,
                       ref_cat_match=refcat_match)

    return results


def check_double_stars(path_or_pixels, autocorr_shape=(2001, 2001), 
                       meas_shape=(201, 201), frac_max_value=1/6.,
                       thresh_func=None, **sextractor_options):
    """
    Check if an image has double stars.

    Parameters
    ----------
    path_or_pixels : str or ndarray
        Full image path or the image pixels as a numpy array.
    autocorr_shape : tuple (optional)
        Cropped image shape for calculating the autocorrelation.
    meas_shape : tuple (optional)
        Measurement image shape for abs(autocorr - flipped).
    frac_max_value : float (optional)
        Fraction of the maximum autocorr diff value to quantify significance.
    thresh_func : function (optional)
        Threshold function = f(N_obj) to decide if an image has double stars. 
    sextractor_options: kwargs (optional)
        Any config option for SExtractor.
        
    Returns
    -------
    val_max : float
        Maximum value of the abs(autocorr diff).
    val_med : float
        Median value of the abs(autocorr diff)
    meas_pixels : ndarray
        Absolute diff of autocorrelation and flipped image.
    has_double_stars : bool
        If True, the image has double stars.
    """
    if thresh_func is None:
        m, b, scatter = 0.0004523, 0.1507, 1.5
        thresh_func = lambda n_obj: (m * n_obj + b) + scatter

    image = utils.load_path_or_pixels(path_or_pixels)
    cropped = improc.crop_image(image, autocorr_shape)

    # TODO: Why not fit a 2D polynomial?
    model = fit_pixels_1d_poly(cropped, order=2, which='rows')
    residual = cropped - model

    cat = sextractor.run(residual, **sextractor_options)
    source_map = create_source_map(cat, residual.shape)
    num_obj = source_map.sum()
    autocorr = signal.fftconvolve(source_map, 
                                  source_map[::-1,::-1], 
                                  mode='same')

    flipped = np.fliplr(autocorr)
    abs_diff = np.abs(autocorr - flipped)
    abs_diff_cropped = improc.crop_image(abs_diff, meas_shape)

    val_max = np.max(abs_diff_cropped)
    val_med = np.median(abs_diff_cropped)
    significance = val_max * frac_max_value

    has_double_stars = significance > thresh_func(num_obj)

    flags = LightFlags()
    if has_double_stars:
        flags.set('DOUBLE_STARS')

    results = DFStruct(flags=flags,
                       val_max=val_max, 
                       val_med=val_med, 
                       abs_acorr_flip_diff=abs_diff_cropped,
                       has_double_stars=has_double_stars)

    return results


def check_image_quality(path_or_pixels, min_num_obj=1000, min_fwhm=1.2, 
                        max_fwhm=5.5, max_ellip=0.25, max_source_asymmetry=0.9, 
                        **sextractor_options):
    """
    Measure number of objects, median object FWHM, and median object 
    ellipticity of image using SExtractor. 

    Parameters
    ----------
    path_or_pixels : str or ndarray
        Full image path or the image pixels as a numpy array.
    **sextractor_options: kwargs (optional)
        Any config option for SExtractor.

    Returns
    -------
    cat : astropy.table.Table
        SExtractor catalog.
    num_obj : int
        Number of objects in the catalog.
    med_fwhm : float
        Median object FWHM.
    med_ellip : float
        Median object ellipticity.
    """
    flags = LightFlags()

    pixels = utils.load_path_or_pixels(path_or_pixels)

    cat = sextractor.run(pixels, 
                         extra_params='ELLIPTICITY', 
                         **sextractor_options)
    num_obj = len(cat)
    med_fwhm = np.median(cat['FWHM_IMAGE'])
    med_ellip = np.median(cat['ELLIPTICITY'])

    # check if sources are piled up on one side
    # which suggests light leaked into the dome (??)
    x_c = pixels.shape[1] / 2
    left = (cat['X_IMAGE'] < x_c).sum()
    right = (cat['X_IMAGE'] > x_c).sum()
    bigger = max(left, right)
    smaller = min(left, right)
    src_asymmetry = 1 - smaller / bigger

    if num_obj < min_num_obj:
        flags.set('TOO_FEW_OBJECTS')
    if med_ellip > max_ellip:
        flags.set('HIGH_ELLIPTICITY')
    if (med_fwhm < min_fwhm) or (med_fwhm > max_fwhm):
        flags.set('BAD_FOCUS')
    if src_asymmetry > max_source_asymmetry:
        flags.set('SOURCE_ASYMMETRY')
    
    results = DFStruct(cat=cat, 
                       flags=flags,
                       num_obj=num_obj, 
                       med_fwhm=med_fwhm, 
                       med_ellip=med_ellip, 
                       src_asymmetry=src_asymmetry)

    return results


def check_pointing(path_or_header, target_radec, max_header_sep=0.5*u.deg, 
                   max_target_sep=60*u.arcmin):
    """
    Check if Dragonfly is pointing in the right place.

    Parameters
    ----------
    path_or_header : str or astropy.io.fits.Header
        Path to fits file or an astropy header.
    target_radec : list-like or astropy.coordinates.SkyCoord
        RA and DEC that we intended to target.
    max_header_sep : astropy Quantity or float
        Max allowed angular separation between the pointing and the RA & DEC in 
        the header. If a float is given, the units are assumed to be degrees.
    max_target_sep : astropy Quantity or float
        Max allowed angular separation between the pointing and the desired 
        target. If a float is given, the units are assumed to be arcmin. 

    Returns
    -------
    flags : dfreduce.LightFlags
        Flag object associated with the given pointing.
    header_sep : astropy Quantity
        Angular separation between the header RA & DEC and the pointing.
    target_sep : astropy Quantity
        Angular separation between the target and pointing.
    """
    header = utils.load_path_or_header(path_or_header)
    if 'WCSAXES' not in header.keys():
        logger.error(f'WCS not found in {path_or_header}')
        return None

    wcs = WCS(header)
    nrows = header['NAXIS2']
    ncols = header['NAXIS1']

    target_sc = utils.to_skycoord(target_radec)
    try:
        head_sc = utils.to_skycoord([header['OBJCTRA'], header['OBJCTDEC']])
    except KeyError:
        head_sc = utils.to_skycoord([header['RA'], header['DEC']])
    pointing_sc = utils.to_skycoord(wcs.all_pix2world(ncols / 2, nrows / 2, 1))

    max_header_sep = utils.check_astropy_units(max_header_sep, 
                                               default_unit='deg')
    max_target_sep = utils.check_astropy_units(max_target_sep, 
                                               default_unit='arcmin')
    on_target = True
    header_sep = pointing_sc.separation(head_sc)
    target_sep = pointing_sc.separation(target_sc)

    results = DFStruct(flags=LightFlags(),
                       header_sep=header_sep,
                       target_sep=target_sep,
                       on_target=on_target)

    if header_sep > max_header_sep:
        on_target = False
        results.flags.set('OFF_HEADER_TARGET')
    if target_sep > max_target_sep:
        on_target = False
        results.flags.set('OFF_TARGET')

    return results


def embed_air_mass(path_or_header, insert_above='COMMENT'):
    header = utils.load_path_or_header(path_or_header)
    coord = header['OBJCTRA'], header['OBJCTDEC']
    skycoord = utils.to_skycoord(coord)
    air_mass = utils.calculate_air_mass(skycoord, header['DATE'])
    bandpass = get_filter_name(header)
    ext_coeff = utils.atm_ext_coeff[bandpass.lower()]
    header.insert(insert_above, ('AIRMASS', air_mass, 'Assumes Hardie (1962)'))
    header.insert(insert_above, ('AIRCOEFF', ext_coeff, 'Extinction coeff'))
    return header


def embed_zp_space(path_or_header, insert_above='COMMENT'):
    header = utils.load_path_or_header(path_or_header)
    assert 'MEDIANZP' in header.keys(), 'MEDIANZP not in header'
    header = embed_air_mass(header)
    zp_space = header['MEDIANZP'] - header['AIRCOEFF'] * header['AIRMASS']
    card = ('ZPSPACE', zp_space, 'The photometric zero point')
    header.insert(insert_above, card)
    return header


def find_halos(serialno_list, zp_list, offset_max=0.2):
    flags = LightFlags()

    fid_zp = [utils.fetch_fiducial_zp(sn) for sn in serialno_list]
    fid_zp = np.array(fid_zp)

    zp_meas = np.asarray(zp_list)
    offsets = fid_zp - zp_meas

    if np.median(np.abs(offsets)) > offset_max:
        flags.set('HALOS')

    results = DFStruct(flags=flags, fiducial_zp=fid_zp)

    return results


def model_sky(path_or_pixels, poly_deg=3,  poly_cross_terms=False, 
              run_label=None, **kwargs):
    """
    Generate sky model using SExtractor and fit a polynomial to it.
    """
    sky_model = sextractor_sky_model(path_or_pixels, 
                                     run_label=run_label, 
                                     **kwargs)
    sky_poly = fit_2d_poly(sky_model, poly_deg, poly_cross_terms)
    results = DFStruct(model=sky_model, 
                       poly=sky_poly.pixels, 
                       coeff=sky_poly.coeff)
    return results

def model_sky_photutils(path_or_pixels, poly_deg=3, box_size=128, 
                        poly_cross_terms=False, mask=None,
                        bkg_estimator='SExtractorBackground', **kwargs):
    """
    Generate sky model using photutils and fit a polynomial to it.
    """
    import photutils 
    pixels = utils.load_path_or_pixels(path_or_pixels)
    if type(bkg_estimator) == str:
        bkg_estimator = getattr(photutils, bkg_estimator)()
    kw = dict(mask=mask, bkg_estimator=bkg_estimator)
    kw.update(kwargs)
    sky_model = photutils.Background2D(pixels, box_size, **kw).background
    sky_poly = fit_2d_poly(sky_model, poly_deg, poly_cross_terms)
    results = DFStruct(model=sky_model, 
                       poly=sky_poly.pixels, 
                       coeff=sky_poly.coeff)
    return results
