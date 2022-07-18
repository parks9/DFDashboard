import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from ..detection import extract_bright_stars
from .. import DFStruct
from .. import logger
from .. import utils
from .. import viz


__all__ = ['check_astrometric_solution']


def check_astrometric_solution(ref_cat, path_or_pixels=None, header=None, 
                               cat=None, cat_cols='X_IMAGE,Y_IMAGE', 
                               ref_cols='ra,dec', max_sep=10, make_plot=False, 
                               ref_cat_frame='icrs', **kwargs):
    """
    Check the quality of the astrometric solution by cross-matching with 
    a reference catalog. 
    """
    if path_or_pixels is not None:
        pixels, header = utils.load_pixels_and_header(path_or_pixels, header)
        cat = extract_bright_stars(path_or_pixels)
    else:
        assert header is not None, 'Must give a header or file path'
        assert cat is not None, 'Must give a SExtractor catalog or file path'
        header = utils.load_path_or_header(header)

    if not utils.has_wcs(header):
        raise Exception('WCS not found')

    # calculate ra and dec taking distortions into account
    wcs = WCS(header)
    cat_cols = utils.list_of_strings(cat_cols)
    ra, dec = wcs.all_pix2world(cat[cat_cols[0]], cat[cat_cols[1]], 1, 
                                ra_dec_order=True)

    ref_cols = utils.list_of_strings(ref_cols)
    if ref_cat_frame.lower() != 'icrs':
        logger.debug('Converting reference catalog coordinate frame to ICRS')
        sc = SkyCoord(ref_cat[ref_cols[0]], ref_cat[ref_cols[1]],
                      unit='deg', frame=ref_cat_frame.lower())
        sc = sc.transform_to('icrs')
        ref_cat[ref_cols[0]] = sc.ra.deg
        ref_cat[ref_cols[1]] = sc.dec.deg

    cat['ra'] = ra
    cat['dec'] = dec
    cat_match, ref_match, sep = utils.match_sky_coordinates(
        cat, ref_cat, cat_cols='ra,dec', ref_cols=ref_cols, sep_max=max_sep)
    
    ra_diff = (ref_match[ref_cols[0]] - cat_match['ra']) * u.deg.to('arcsec')
    ra_diff *= np.cos(np.deg2rad(cat_match['dec']))
    dec_diff = (ref_match[ref_cols[1]] - cat_match['dec']) * u.deg.to('arcsec')

    nx = header['NAXIS1']
    ny = header['NAXIS2']
    dx = cat_match['X_IMAGE'] - (nx / 2)
    dy = cat_match['Y_IMAGE'] - (ny / 2)
    dr_image =  np.sqrt(dx**2 + dy**2)

    results = DFStruct(delta_ra=ra_diff, 
                       delta_dec=dec_diff, 
                       angular_sep=sep.arcsec,
                       dr_image=dr_image, 
                       cat_match=cat_match, 
                       ref_match=ref_match, 
                       header=header)

    if make_plot:
        fig, ax = viz.wcs_quality_check(results, **kwargs)
        results['fig'] = fig
        results['ax'] = ax
    
    return results
