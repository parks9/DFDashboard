# Standard library
import os
from copy import deepcopy

# Third-party
import numpy as np
import pandas as pd
from scipy import ndimage
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
from reproject import reproject_interp
from fitsio import read_header as fitsio_read

# Project
from .. import utils, apass_path, logger
from .. import DFStruct, improc, astrometry
from ..utils import log_header_label
from ..cameras import get_filter_name
from ..tasks import lights as tasks
from ..flags import LightFlags, FlagArray


__all__ = [ 
    'assess_quality', 
    'embed_astrometry', 
    'model_and_subtract_sky', 
    'embed_zero_points',
    'process_light',
    'reject_aberrant_zero_points',
    'process_light_deep'
]

def log_flag(fn, flag_string):
    msg = f'Skipping {fn} with flags: '
    logger.warning(msg + flag_string)


def check_path_table(path_table):
    # check type 
    if type(path_table) == pd.DataFrame:
        if path_table.index.name != 'light_id':
            logger.warning('Index should be the light ID --> changing it')
            path_table.index.name = 'light_id'
    elif type(path_table) == Table:
        for n in ['light', 'master_dark', 'master_flat']:
            assert n in path_table.colnames, f'must have these columns {names}'
        if 'light_id' not in path_table.colnames:
            logger.warning('Light ID not provided --> using array positions')
            path_table['light_id'] = np.arange(len(path_table))
        path_table = path_table.to_pandas(index='light_id')
    else:
        raise Exception(f'{type(path_table)} is not a valid path_table type')
    # add serialno and expnum if necessary
    if 'serialno' not in path_table.columns:
        logger.debug('Assuming default file name structure to get serialno')
        path_table['serialno'] = path_table.light.\
            apply(lambda s: s.split('/')[-1].split('_')[0])
    if 'expnum' not in path_table.columns:
        logger.debug('Assuming default file name structure to get expnum')
        path_table['expnum'] = path_table.light.\
            apply(lambda s: int(s.split('/')[-1].split('_')[1]))
    if 'filter_name' not in path_table.columns:
        logger.debug('Using headers to add filter_name to path table')
        filter_names = []
        for fn in path_table.light:
            filter_names.append(get_filter_name(fn))
    return path_table


def force_time_consensus(path_table):
    # force time consensus 
    # TODO: this should happen when we create the database!
    path_table['date'] = pd.NA
    groups = path_table.groupby('expnum').groups
    for expnum, idx in groups.items():
        paths = path_table.loc[idx].light.values
        times = utils.find_time_consensus(paths)
        path_table.loc[idx, 'date'] = times.date.to_value('isot')
        num_off_time = times.off_time.sum()
        if num_off_time > 0:
            msg = f'{num_off_time} camera(s) off time for expnum {expnum}'
            logger.warning(msg + ' --> will force consensus')
            update_idx = idx[times.off_time]
            path_table.loc[update_idx, 'date'] = times.mode.to_value('isot')
    return path_table


def assess_quality(light_path, dark_path, flat_path, 
                   run_label='light', config={}):
    flags = LightFlags()
    _c = deepcopy(config)
    light = improc.subtract_master_dark(light_path, dark_path)

    flat_kw = _c.pop('divide_by_master_flat', {})
    light = improc.divide_by_master_flat(light.pixels, 
     				         flat_path, 
					 light.header, 
                                         **flat_kw) 

    log_label = log_header_label(light.header)

    double_stars_kw = _c.pop('check_double_stars', {}) 
    logger.debug(f'Looking for double stars: ' + log_label)
    flags += tasks.check_double_stars(light.pixels, 
                                      run_label=run_label,
                                      **double_stars_kw).flags

    image_quality_kw = _c.pop('check_image_quality', {})
    logger.debug(f'Checking image quality: ' + log_label)
    flags += tasks.check_image_quality(light.pixels, 
                                       run_label=run_label, 
                                       **image_quality_kw).flags

    light['flags'] = flags
    light['path'] = light_path
    logger.debug(f'{flags.count()} flags for {log_label}')

    return light


def embed_astrometry(light, run_label='light', config={}):
    _c = deepcopy(config)
    kw = _c.pop('solve_field', {})
    logger.debug(f'Solving field: ' + log_header_label(light.header))
    astrom = astrometry.solve_field(light.pixels,
                                    light.header,
                                    run_label=run_label, 
                                    overwrite=True, 
                                    target_radec=light.target_radec,
                                    **kw)
    if not astrom.success:
        light.flags.set('ASTROMETRY_FAILED')
        log_flag(light.path, light.flags.to_string())
        return light

    light['astrom'] = astrom
    light['header'] = astrom.header
    
    return light


def model_and_subtract_sky(light, run_label='light', **kwargs):
    logger.debug(f'Modeling sky background: {log_header_label(light.header)}')

    insert_above = kwargs.pop('insert_above', 'COMMENT')
    sky = tasks.model_sky(light.pixels, run_label=run_label, **kwargs)
    light['sky'] = sky

    light.pixels = light.pixels - sky.poly

    card = ('MEDSKY', np.median(sky.poly), 'median sky value')
    light.header.insert(insert_above, card)

    for i in range(sky.coeff.shape[0]):
        for j in range(sky.coeff.shape[1]):
            card = (f'SKY_C{i}{j}', sky.coeff[i, j], 'sky coeff')
            light.header.insert(insert_above, card)

    return light


def register_image(light, **kwargs):
    shape = kwargs.pop('shape', (4250, 5750))
    pixscale = kwargs.pop('pixscale', 2.5)
    
    ra, dec = light.target_radec.ra.deg, light.target_radec.dec.deg
    
    logger.debug(f'Registering image: {log_header_label(light.header)}')
    resampled = improc.resample_image(
	light.pixels, ra, dec, pixscale, int(shape[1]), int(shape[0]),
	light.astrom.fitsio_header)

    # update header with new wcs
    header = astrometry.remove_previous_wcs(light.header)
    for k, v in resampled.wcs.to_header(relax=True).items():
        header[k] = v

    insert_above = kwargs.pop('insert_above', 'COMMENT')
    header.insert(insert_above, ('PIXSCALE', pixscale, 'arcsec / pixel'))
    header = utils.update_header(header, 'Registered')

    light = DFStruct(pixels=resampled.pixels, 
                     header=header, 
                     astrom=light.astrom,
                     flags=light.flags,
                     target_radec=light.target_radec,
                     base_fn=os.path.basename(light.path))
                     
    return light


def embed_zero_points(light, ref_cat_or_path, run_label='light', 
                      insert_above='COMMENT', config={}):
    
    if type(ref_cat_or_path) == str:
        cat = 'APASS'
        ref_path = ref_cat_or_path
    else:
        cat = ref_cat_or_path
        ref_path = None

    _c = deepcopy(config)
    calc_zp_kw = _c.pop('calculate_zp', {})

    logger.debug(f'Calculating zero point: {log_header_label(light.header)}')
    zp_info = tasks.calculate_zp(light.pixels, 
                                 get_filter_name(light.header), 
                                 catalogue=cat,
                                 header=light.header, 
                                 catalogue_dir=ref_path, 
                                 run_label=run_label,
                                 **calc_zp_kw)
    
    if zp_info is None:
        light.flags.set('TOO_FEW_REF_SOURCES')
        results = DFStruct(light=light, ref_cat=None)
        return results

    card = ('MEDIANZP', zp_info.zp_median, 'Median mag offset from APASS')
    light.header.insert(insert_above, card)

    light.header = tasks.embed_zp_space(light.header,   
                                        insert_above=insert_above)
    fid_zp = utils.fetch_fiducial_zp(light.header['SERIALNO'])
    card = ('FIDZP', fid_zp, 'Fiducial zero point of camera')
    light.header.insert(insert_above, card)
    results = DFStruct(light=light, 
                       img_cat_match=zp_info.img_cat_match,
                       ref_cat_match=zp_info.ref_cat_match)

    return results


def process_light(target_radec, light_path, dark_path, flat_path, ref_cat, 
                  out_path, sip_head_path=None, run_label='light', 
                  config={}, date=None):

    _c = deepcopy(config)

    # dark subtract and flatten light
    light = improc.subtract_master_dark(light_path, dark_path)
    flat_kw = _c.pop('divide_by_master_flat', {})
    light = improc.divide_by_master_flat(light.pixels, 
                                         flat_path, 
                                         light.header, 
                                         **flat_kw) 
    light['flags'] = LightFlags()
    light['path'] = light_path
    light['target_radec'] = target_radec

    # update header with consensus time (if necessary)
    if date is not None:
        light.header['DATE'] = date, '(YYYY-MM-DDThh:mm:ss UTC)'

    # find astrometric solution 
    logger.debug(f'Solving field for {run_label}.')
    solve_field_kw = _c.pop('solve_field', {})
    astrom = astrometry.solve_field(
        light.pixels, 
        light.header, 
        target_radec=target_radec,
        run_label=run_label, 
        overwrite=True,
        **solve_field_kw
    )
    if not astrom.success:
        light.flags.set('ASTROMETRY_FAILED')
        log_flag(light.path, light.flags.to_string())
        results = DFStruct(light=light, ref_cat=None, path=None, 
                           flags=light.flags, success=False)
        return results

    light['astrom'] = astrom
    light['header'] = astrom.header

    if sip_head_path is not None:
        fn = os.path.basename(light_path.replace('.fits', '.head'))
        fn = os.path.join(sip_head_path, fn)
        logger.debug(f'Writing header to {fn}.')
        astrom.header.tofile(fn, overwrite=True)

    # check if pointing is on target
    pointing = tasks.check_pointing(light.astrom.header, light.target_radec)
    if not pointing.on_target:
        logger.warning(f'Off target: {log_header_label(light.header)}')
        light.flags += pointing.flags
        results = DFStruct(light=light, ref_cat=None, path=None, 
                           flags=light.flags, success=False)
        return results

    # model sky background and subtract from light
    light = model_and_subtract_sky(light, run_label, **_c['model_sky'])

    # resample light to common pixel grid
    light = register_image(light, **_c['register_image'])

    # measure and embed zero point 
    results = embed_zero_points(
        light, ref_cat, run_label=run_label, config=_c)
    results['flags'] = results.light.flags
    results['success'] = results.light.flags.count() == 0 

    # write processed light
    if results.success:
        base_fn = light.base_fn.replace('.fits', '_reg.fits')
        out_fn = os.path.join(out_path, base_fn)
        utils.write_pixels(out_fn, light.pixels, light.header)
        results['path'] = out_fn

    return results


def reject_aberrant_zero_points(path_table, reg_out_path, suffix='reg',
                                max_zp_offset=0.25, target=None):

    target_name = '.' if target is None else f' from field {target}.'
    logger.info('Rejecting frames with aberrant zero points' + target_name)
    path_table = check_path_table(path_table)
    flag_arr = FlagArray('light', index=path_table.index)

    fid_zp = {}
    meas_zp = {}
    idx_zp = {}
    diff_zp = {}

    # group by exposure number for zp comparison
    groups = path_table.groupby('expnum').groups
    for expnum, group_idx in groups.items():

        # params for fixed exposure time
        fid_zp[expnum] = []
        meas_zp[expnum] = []
        idx_zp[expnum] = []
        diff_zp[expnum] = []

        for light_id, row in path_table.loc[group_idx].iterrows():
            fn = f'{row.serialno}_{row.expnum}_light_{suffix}.fits'
            fn = os.path.join(reg_out_path, fn)
            header = fits.getheader(fn)

            idx_zp[expnum].append(light_id)
            fid_zp[expnum].append(header['FIDZP'])
            meas_zp[expnum].append(header['MEDIANZP'])

        # compare zero points with fiducial values at fixed exposure time
        if len(fid_zp[expnum]) > 0:
            diff = np.array(fid_zp[expnum]) - np.array(meas_zp[expnum])
            if np.median(np.abs(diff)) > max_zp_offset:
                flag_arr.set(idx_zp[expnum], 'HALOS')
            diff_zp[expnum] = diff

    results = DFStruct(flag_arr=flag_arr, zp_offsets=diff_zp)
    
    return results


def process_light_deep(target_radec, light_path, dark_path, flat_path, ref_cat, 
                       out_path, deep_coadd_pixels, deep_coadd_header, 
                       sip_head_path=None, run_label='light', config={}):

    _c = deepcopy(config)

    # dark subtract and flatten light
    light = improc.subtract_master_dark(light_path, dark_path)
    flat_kw = _c.pop('divide_by_master_flat', {})
    light = improc.divide_by_master_flat(light.pixels, 
                                         flat_path, 
                                         light.header, 
                                         **flat_kw) 
    light['flags'] = LightFlags()
    light['path'] = light_path
    light['target_radec'] = target_radec

    # find astrometric solution 
    needs_astrometry = True
    if sip_head_path is not None:
        fn = os.path.basename(light_path.replace('.fits', '.head'))
        fn = os.path.join(sip_head_path, fn)
        if os.path.isfile(fn):
            logger.debug(f'Loading header from {fn}')
            light['header'] = fits.Header.fromfile(fn)
            light['astrom'] = DFStruct(fitsio_header=fitsio_read(fn))
            needs_astrometry = False
        else:
            logger.warning(f'Header for {fn} not found.')
    if needs_astrometry: 
        logger.debug(f'Solving field for {run_label}.')
        solve_field_kw = _c.pop('solve_field', {})
        astrom = astrometry.solve_field(
            light.pixels, 
            light.header, 
            target_radec=target_radec,
            run_label=run_label, 
            overwrite=True,
            **solve_field_kw
        )
        light['astrom'] = astrom
        light['header'] = astrom.header

    # project deep coadd into light pixel grid
    logger.debug(f'Reprojecting deep coadd into {run_label} pixel grid.')
    coadd_reproj = reproject_interp((deep_coadd_pixels, deep_coadd_header), 
                                    light.header, return_footprint=False)

    # TODO: I skipped the uniform_filter step. Can we improve this?
    isfinite = np.isfinite(coadd_reproj)
    positive = np.greater(coadd_reproj, 0, where=isfinite)
    floor = np.nanmedian(coadd_reproj[positive])
    mask = np.greater(coadd_reproj, floor, where=isfinite).astype(float)
    mask = ndimage.median_filter(mask, size=_c['deep_mask_filter_size'])
    weight_map = (mask == 0).astype('float')
    wmap_fn, _ = utils.temp_fits_file(weight_map, run_label=run_label)

    # model sky background and subtract from light
    kw = _c['model_sky']
    kw['DETECT_THRESH'] = 2
    kw['WEIGHT_IMAGE'] = wmap_fn
    kw['WEIGHT_TYPE'] = 'MAP_WEIGHT'
    light = model_and_subtract_sky(light, run_label, **kw)

    # resample light to common pixel grid
    light = register_image(light, **_c['register_image'])

    # measure and embed zero point 
    results = embed_zero_points(light, ref_cat, run_label=run_label, config=_c)
    results['weight_map'] = weight_map
    results['flags'] = results.light.flags
    results['success'] = results.light.flags.count() == 0 

    # write processed light
    if results.success:
        base_fn = light.base_fn.replace('.fits', '_reg_deep.fits')
        out_fn = os.path.join(out_path, base_fn)
        utils.write_pixels(out_fn, light.pixels, light.header)
        results['path'] = out_fn

    # clean up 
    os.remove(wmap_fn)

    return results
