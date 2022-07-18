import os
import subprocess
import numpy as np
from scipy import optimize
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from ..log import logger
from .. import DFStruct
from .. import utils
from .pam import pixel_area_map


__all__ = ['remove_previous_wcs', 'TweakWCS', 'solve_field']


try:
    import fitsio
    FOUND_FITSIO = True
except:
    FOUND_FITSIO = False


def remove_previous_wcs(header):
    header = utils.load_path_or_header(header)
    if utils.has_wcs(header):
        header_str = header.tostring()
        astrom_start = header_str.find('COMMENT Original key:') 
        end_str = '--(Put in by the new-wcs program)--'
        astrom_end = header_str.find(end_str)
        astrom_end += header_str[astrom_end:].find('COMMENT') 
        num_spaces_to_next_comment = 73
        astrom_end += len('COMMENT') + num_spaces_to_next_comment
        new_str = header_str[:astrom_start] + header_str[astrom_end:]
        header = header.fromstring(new_str)
        header = utils.update_header(header, 'Previous ASTROM overwritten')
    else:
        logger.warning('No WCS found --> will not rewrite')
    return header


def rotation_matrix(angle_deg):
    sin_phi = np.sin(np.deg2rad(angle_deg))
    cos_phi = np.cos(np.deg2rad(angle_deg))
    R = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
    return R


class TweakWCS(object):

    def __init__(self, header, cat_radec, ref_radec, cat_cols=['ra','dec'],
                 ref_cols=['ra','dec'], optimize_func=optimize.fmin):
        self.wcs = WCS(header)
        xy = self.wcs.all_world2pix(cat_radec[cat_cols[0]],
                                    cat_radec[cat_cols[1]], 1)
        self.cat_pix = np.array(xy).flatten()
        self.ref_ra = ref_radec[ref_cols[0]]
        self.ref_dec = ref_radec[ref_cols[1]]
        self.crval = self.wcs.wcs.crval.copy()
        self.cd = self.wcs.wcs.cd.copy()
        self.header = header.copy()
        self.optimize_func = optimize_func

    def optimize(self, **kwargs):
        p0 = np.array([0, 0, 0])
        self.results = self.optimize_func(self, p0, **kwargs)
        for k, v in self.wcs.to_header(relax=True).items():
            self.header[k] = v
        return self.header

    def update_header(self, header, inplace=True):
        h = header if inplace else header.copy()
        for k, v in self.header.items():
            h[k] = v
        if not inplace:
            return h

    def __call__(self, params):
        d_ra, d_dec, d_theta = params
        d_radec = np.array([d_ra, d_dec]) * u.arcsec.to('deg')
        self.wcs.wcs.crval = self.crval + d_radec
        self.wcs.wcs.cd = np.dot(rotation_matrix(d_theta), self.cd)
        ref_xy = self.wcs.all_world2pix(self.ref_ra, self.ref_dec, 1)
        ref_pix = np.array(ref_xy).flatten()
        return np.sum((self.cat_pix - ref_pix)**2)


def solve_field(path_or_pixels, header=None, run_label=None, clean=True,
                save_objs_plot=False, pixscale_limits=[2.7, 2.9], 
                tmp_path='/tmp', overwrite=False, sip_order=3, 
                max_num_objects=3000, downsample=4, target_radec=None, 
                search_radius=3, options=None, index_path=None, 
                config_fn=None, identifier='serialno', fov_limits=[1, 3]):
    """
    Solve field using astrometry.net.

    Parameters
    ----------
    path_or_pixels : str or ndarray
        Full image path or the image pixels as a numpy array.
    header : astropy.io.fits.header.Header 
        Image header. If you pass image pixels, you must also give the header.
    run_label : str (optional)
        A unique label for the temporary files.
    clean : bool (optional)
        If True, delete all extra files created by astrometry.net.
    save_objs_plot : bool (optional)
        If True, astrometry.net will (slowly) create plots.
    pixscale_limits : list of floats
        Max and min pixel scale: [scale low, scale high]
    fov_limits : list of floats (optional)
        Max and min FOV size: [low, high]. Units are degrees.
        To use set pixscale_limits to None.
    tmp_path : str (optional)
        Path for temporary fits files if you give image pixels as input.
    overwrite : bool (optional)
        If True, solve field even if the header already has a WCS and overwrite
        overwrite the header.
    sip_order : int (optional)
        Order of the Simple Image Polynomials (SIP)
    max_num_objects : int (optional)
        Max number of sources (after sorting) for astrometry.net to consider.
    downsample : int (optional)
        Factor to downsample the image before running SExtractor.
    target_radec : list (floats or strings) or SkyCoord (optional)
        Only search index files near this position.
    search_radius: float (optional)
        If target_radec is given, search within this radius in degrees. 
    options : str (optional)
        Additional options for solve-field given as you would enter them 
        in the terminal. (e.g., options='--dir my_output_dir')
    index_path : str (optional)
        Path to the index files. If not None, a temporary config file will be 
        created (this seems to be the only way to pass this option). If None, 
        the default astrometry.net config file will be used.
    config_fn : str (optional)
        Configuration file name for astrometry.net solve-field. If not None, 
        the index_path will be ignored. 
        
    Returns
    -------
    header : astropy.fits.Header
        Astropy header with the WCS embedded.
    fitsio_header : fitsio.header.FITSHDR
        Header from fitsio (if it is installed). This is the header format that 
        is required for resampling images using the astrometry.net python code.
    pam : ndarray
        Pixel area map normalized to equal one at 1 at the largest pixel.
    success : bool
        Will be True if the WCS was (thought to be) successfully solved.
    """
    pixels, header = utils.load_pixels_and_header(path_or_pixels, header)
    label = '' if run_label is None else '-' + run_label

    if identifier is not None:
        id_name = header[identifier].strip()
    else:
        id_name = 'Unknown'

    if 'WCSAXES' in header.keys():
        if overwrite:
            logger.warning(f'Overwriting the WCS of {id_name}{label}')
            header = remove_previous_wcs(header)
        else:
            msg = f'The image for {id_name}{label} already has a WCS! '
            msg += 'Use overwrite = True to overwrite.'
            raise Exception(msg)

    temp_image_path, created_tmp = utils.temp_fits_file(
        pixels, tmp_path, run_label, id_name, header)

    scaletype = 'arcsecperpix' if pixscale_limits is not None else 'degwidth'
    limits = pixscale_limits if pixscale_limits is not None else fov_limits
    cmd = f'solve-field --cpulimit 30 -u {scaletype} --fits-image --overwrite'
    cmd = cmd + ' --tweak-order {} --objs {} -L {} -H {}'
    cmd = cmd.format(sip_order, max_num_objects, *limits)

    prefix = temp_image_path.replace('.fits', '')
    cmd += f' --new-fits {prefix}_wcs.fits'

    if not save_objs_plot:
        cmd += ' --no-plots'
    if downsample is not None:
        if downsample > 1:
            cmd += f' --downsample {downsample}'

    if target_radec is not None:
        target_sc = utils.to_skycoord(target_radec)
        ra, dec = target_sc.ra.deg, target_sc.dec.deg
        cmd += f' --ra {ra} --dec {dec} --radius {search_radius}'

    if options is not None:
        cmd += ' ' + options 

    if config_fn is None and index_path is not None:
        config_fn = os.path.join(tmp_path, f'astrometry{label}.cfg')
        with open(config_fn, 'w') as file:
            print(f'add_path {index_path}\nautoindex', file=file)
        cmd += f' --config {config_fn}'
        tmp_config = True
    else:
        tmp_config = False
 
    cmd += ' ' + temp_image_path

    success = False
    base_fn = os.path.basename(temp_image_path)
    logger.debug('Solving field for ' + base_fn)
    logger.debug(f'>>> {cmd}')
    subprocess.call(cmd, shell=True)

    if os.path.isfile(prefix + '.solved'):
        success = True
        logger.debug('Successfully solved field for ' + base_fn)
        _header = fits.getheader(prefix + '_wcs.fits')
        if FOUND_FITSIO:
            fitsio_header = fitsio.read_header(prefix + '_wcs.fits')
        else:
            fitsio_header = None
        _header['EPOCH'] = 2000.0
        pam = pixel_area_map(_header, normalize_at='peak')
        results = DFStruct(header=_header, fitsio_header=fitsio_header, 
                           pam=pam, success=success)
    else:
        logger.warning('Astrometry failed for ' + base_fn)
        results = DFStruct(success=success)

    if clean:
        if success:
            clean_ext = ['.rdls', '-indx.xyls', '.axy', '.wcs',
                         '.corr', '.match', '.solved']
            if save_objs_plot:
                clean_ext.extend(['-ngc.png', '-indx.png'])
            for ext in clean_ext:
                fn = prefix + ext
                logger.debug('Deleting ' + fn)
                os.remove(fn)
        else:
            os.remove(prefix + '.axy')
    if created_tmp and clean:
        os.remove(temp_image_path)
        if success:
            os.remove(prefix + '_wcs.fits')
    if tmp_config and clean:
        os.remove(config_fn)

    return results
