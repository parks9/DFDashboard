import os
import numpy as np
from astropy import units as u
from astropy.coordinates import Longitude, match_coordinates_sky, SkyCoord
from astropy.table import Table, vstack
from .. import package_dir, utils, logger


__all__ = [
    'getminmaxra',
    'get_apass_filename', 
    'load_apass_from_file',
    'load_apass_in_region',
    'match_catalogues',
    'match_sky_coordinates'
]


# default APASS columns to exclude
exclude_names = ['raerr','decerr','nobs','mobs','V','BV','B','i','Verr',
                 'BVerr','Berr','ierr']


def getminmaxra(ra_array,buffer=1.1):
    """
    Return the minimum RA (minus the buffer) and the maximum RA (plus the 
    buffer) from the given RA array input, within the range 0 to 360 deg (wraps
    around 360 deg to 0 deg).

    Parameters
    ----------
    ra_array : array
        An array of RA values (in degrees).
    buffer : float 
        Value to add/subtract to the minimum/maximum RA value.

    Returns
    -------
    minra : float
        The minimum RA in the input array, minus the buffer value.
    maxra : float
        The maximum RA in the input array, plus the buffer value.
    """
    minra,maxra = Longitude([np.min(ra_array)-buffer,
                             np.max(ra_array)+buffer],unit='deg').value
    if (np.abs(maxra-minra) > 180.):
        ra_wrap = Longitude(ra_array,unit='deg',wrap_angle=180*u.deg)
        minra,maxra = Longitude([min(ra_wrap).value-buffer,
                                 max(ra_wrap).value+buffer],unit='deg').value
    return minra,maxra


def get_apass_filename(dec,apass_dir):
    """
    Return an APASS catalogue filename, corresponding to the given declination.
    
    Parameters
    ----------
    dec : float
        A declination value (in degrees).
    apass_dir : directory
        The path to the directory where the APASS files are stored.

    Returns
    -------
    apass_fn : string
        The APASS catalogue filename with path.
    """
    
    if dec > 0:
        dec_label = str(int(5*int(np.float(dec)/5))).zfill(2)
    else:
        dec_label = str(int(5*np.ceil(np.float(np.abs(dec)/5)))).zfill(2)

    v_label = 8 if dec >= 20 else 9
    pm_label = 'm' if dec < 0 else 'p'
    apass_fn = f'z{pm_label}{dec_label}_{v_label}.sum'
    apass_fn = os.path.join(apass_dir,apass_fn)

    if os.path.isfile(apass_fn):
        return apass_fn
    else:
        logger.error(f'No appropriate APASS catalog file ({apass_fn}) was found.')
        return None


def load_apass_from_file(apass_fn,minra=None,maxra=None,
                         exclude_names=exclude_names):
    """
    Loads the APASS catalogue from a file (with option to select out 
    a range in RA.
    
    Parameters
    ----------
    apass_fn : string
        Name of filename containing APASS catalogue to be loaded.
    minra : float (optional)
        Minimum RA value (in degrees) to keep.
    maxra : float (optional)
        Maximum RA value (in degrees) to keep.
    exclude_names : list of strings
        List of columns to exclude when loading APASS file.

    Returns
    -------
    apass : astropy Table
        The APASS reference catalogue.
    """
    logger.info(f'Loading APASS catalogue from file {apass_fn}.')
    col_names = ['name','ra','raerr','dec','decerr','nobs','mobs','V','BV','B',
                 'g','r','i','Verr','BVerr','Berr','gerr','rerr','ierr']
    dtype = ['int64'] + ['float64'] * 4 + ['int64'] * 2 + ['float64'] * 12

    # np.loadtxt is faster than Table.read
    exclude_idx = [col_names.index(n) for n in exclude_names]
    use_idx = [i for i in range(len(col_names)) if i not in exclude_idx]
    data = np.loadtxt(apass_fn, usecols=use_idx)

    # store apass catalog in an astropy table
    names = [col_names[i] for i in use_idx]
    dtype = [dtype[i] for i in use_idx]
    apass = Table(data, names=names, dtype=dtype)

    if (minra is not None) and (maxra is not None):
        logger.debug(f'Selecting sources between RA = {minra:.1f} and {maxra:.1f}.')
        if (np.abs(maxra - minra) > 180):
            mask = (apass['ra'] < maxra) | (apass['ra'] > minra)
        else:
            mask = (apass['ra'] < maxra) & (apass['ra'] > minra)
        apass = apass[mask]
        
    logger.debug(f'Found {len(apass)} sources in APASS.')
    
    return apass

def load_apass_in_region(apass_dir,cat=None,bounds=None):
    """
    Loads APASS reference catalogue based on SExtractor catalogue or bounds in
    RA and Dec (degrees). Will load up to two APASS files (up to 10 deg 
    coverage in Dec).
    
    Parameters
    ----------
    apass_dir : directory
        The path to the directory where the APASS files are stored.
    cat : SExtractor catalogue (optional)
        SExtractor catalogue over some region in the sky.  Used to determine
        RA and Dec region to load from the APASS catalogue.
        Must supply cat if don't supply bounds.
    bounds : array (optional)
        Array containing bounds of region to load APASS catalogue within, in 
        format: [mindec, maxdec, minra, maxra].
        Must supply bounds if don't supply cat.

    Returns
    -------
    apass : astropy Table
        The APASS reference catalogue.
    """
    if bounds is None and cat is None:
        logger.error('load_apass_in_region: No bounds or SExtractor cat given '+
                     '- need to supply one or the other.')
        return None
    if bounds is None:
        mindec,maxdec = np.min(cat['Y_WORLD'])-1.1,np.max(cat['Y_WORLD'])+1.1
        minra,maxra = getminmaxra(cat['X_WORLD'])
    else:
        if len(bounds)!=4:
            logger.error('load_apass_in_region: Bounds not in correct format. '+
                         'Should be array with four items: '+
                         '[mindec, maxdec, minra, maxra].')
            return None
        mindec,maxdec,minra,maxra = bounds
        logger.info('Loading APASS between ' 
                    f'Dec = {mindec:.2f} to {maxdec:.2f} and '+
                    f'RA = {minra:.2f} to {maxra:.2f}.')

    apass_fn_mindec = get_apass_filename(mindec,apass_dir)
    apass = load_apass_from_file(apass_fn_mindec,minra=minra,maxra=maxra)

    apass_fn_maxdec = get_apass_filename(maxdec,apass_dir)
    if not (apass_fn_mindec == apass_fn_maxdec):
        apass_maxdec = load_apass_from_file(apass_fn_maxdec,minra=minra,maxra=maxra)
        apass = vstack([apass,apass_maxdec])

    logger.debug(f'Selecting data between Dec = {mindec:.1f} and {maxdec:.1f}.')
    mask = (apass['dec'] < maxdec) & (apass['dec'] > mindec)
    apass = apass[mask]
    
    logger.info(f'Found {len(apass)} sources from APASS in region.')

    return apass


def match_catalogues(cat,ref,filter,sep_max=2.0,mag_min=14.0,mag_max=16.5):
    """
    Match sources in a SourceExtractor catalogue to a reference catalogue (in 
    astropy Table format) by RA and Dec.

    Parameters
    ----------
    cat : SExtractor catalogue
        The output catalogue from running SExtractor on an image.
        Needs to have columns FLUX_AUTO, FLUXERR_AUTO, X_WORLD, Y_WORLD
    ref : astropy Table 
        A reference catalogue with sources covering same region as image.
        Needs to have columns with same name as input filter, 'ra' and 'dec'.
    filter: str 
        Name of the filter used in image (needs to be in reference catalogue).
    sep_max : float (optional)
        Maximum allowable star separation between image and
        reference catalogue (arcsec). Default is 2.0 arcsec.
    mag_min : float (optional)
        Exclude sources brighter than this from analysis. Default is 14.0 mag.
    mag_max : float (optional)
        Exclude sources fainter than this from analysis. Default is 16.5 mag.

    Returns
    -------
    cat_match : astropy Table
        Matched image catalogue
    ref_match : astropy Table
        Matched reference catalogue
    """
    
    if filter not in ref.columns:
        logger.error(f'Desired filter "{filter}" not in reference catalogue.')
        logger.debug('Reference catalogue columns:')
        logger.debug(ref.columns)
        return None

    ref_cut = ref[filter] > mag_min
    ref_cut &= ref[filter] < mag_max
    if ('g' in ref.columns) & ('r' in ref.columns):
        ref_cut &= (ref['g'] - ref['r']) > -5
        ref_cut &= (ref['g'] - ref['r']) < 5
    if 'gerr' in ref.columns:
        ref_cut &= ref['gerr'] != 0.
    if 'rerr' in ref.columns:
        ref_cut &= ref['rerr'] != 0.

    im_cut = cat['FLUX_AUTO'] > 0
    
    ref = ref[ref_cut]
    cat = cat[im_cut]

    cat_match, ref_match, _ = match_sky_coordinates(
        cat, ref, ['X_WORLD', 'Y_WORLD'], ['ra', 'dec'], sep_max)
    
    cat_match[filter] = -2.5*np.log10(cat_match['FLUX_AUTO'])
    cat_match[f'{filter}_err'] = cat_match['FLUXERR_AUTO']
    cat_match[f'{filter}_err'] /= np.log(10)*cat_match['FLUX_AUTO']

    return cat_match,ref_match


def match_sky_coordinates(cat, ref_cat, cat_cols='ra,dec',
                          ref_cols='ra,dec', sep_max=2.0):
    """
    Match two catalogs based on their sky coordinates.

    Parameters
    ----------
    cat : astropy.table.Table
        Catalog you want to match.
    ref_cat : astropy.table.Table
        Reference catalog to match 'cat' with.  
    cat_cols : list or str (optional)
        Column names of 'cat' corresponding to RA and Dec. Will assume 
        SExtractor names by default.
    ref_cols : list or str (optional)
        Column names of 'ref_cat' corresponding to RA and Dec.
    sep_max : float (optional)
        Maximum allowable star separation between image and
        reference catalogue (arcsec). Default is 2.0 arcsec.

    Returns
    -------
    cat_match : astropy.table.Table
        Matched image catalog.
    ref_match : astropy.table.Table
        Matched reference catalog.
    sep : astropy.units.Quantity
        Angular separation between the matched sources.

    Notes
    -----
    The returned catalogs will be the same length, with each row corresponing 
    to the (hopefully same) matched object.
    """
    cat_cols = utils.list_of_strings(cat_cols)
    ref_cols = utils.list_of_strings(ref_cols)
    cat_sc = SkyCoord(cat[cat_cols[0]], cat[cat_cols[1]], unit='deg')
    ref_sc = SkyCoord(ref_cat[ref_cols[0]], ref_cat[ref_cols[1]], unit='deg')
    idx, sep, _ = match_coordinates_sky(cat_sc, ref_sc)
    match = sep.arcsec < sep_max
    cat_match = cat[match].copy()
    ref_match = ref_cat[idx[match]].copy()
    sep = sep[match]
    return cat_match, ref_match, sep
