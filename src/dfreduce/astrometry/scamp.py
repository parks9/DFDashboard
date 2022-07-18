import os
from subprocess import call
from astropy.io import fits
from ..detection import sextractor
from .. import DFStruct
from .. import package_dir
from .. import utils, logger


input_file_path = os.path.join(package_dir, 'astrometry/scamp_files')
default_config = os.path.join(input_file_path, 'default.config')


# get list of all config options
file = open(os.path.join(input_file_path, 'config_options.txt'), 'r')
all_option_names = [line.rstrip() for line in file]
file.close()


# default non-standard options
default_options = dict(
    MAGZERO_KEY='ZP',
    PHOTINSTRU_KEY='TARGET',
    PROJECTION_TYPE='TAN',
    ASTREFMAG_LIMITS='13,18',
    ASTRINSTRU_KEY='SERIALNO',
    STABILITY_TYPE='EXPOSURE',
    sextractor=dict(ANALYSIS_THRESH=10, DETECT_THRESH=10)
)



def run(path_or_pixels, header=None, tmp_path='/tmp', run_label=None, 
        catalog_path=None, se_kw={}, executable='scamp', 
        config_path=default_config, save_scamp_header=False, **scamp_options):

    image_path, created_tmp = utils.temp_fits_file(path_or_pixels,
                                                   tmp_path=tmp_path,
                                                   run_label=run_label,
                                                   prefix='scamp_tmp',
                                                   header=header)
    header = utils.load_path_or_header(image_path)
    if 'WCSAXES' not in header.keys():
        raise Exception(f"No WCS: {header['SERIALNO']}-{header['TARGET']}")

    if catalog_path is not None:
        cat_name = catalog_path
        save_cat = True
    else:
        label = '' if run_label is None else '_' + run_label
        cat_name = os.path.join(tmp_path, f'se{label}.cat')
        save_cat = False
    
    # setup sextrator options and run it
    kw = default_options.pop('sextractor')
    for k, v in se_kw.items():
        if k == 'CATALOG_TYPE' or k == 'extra_params':
            logger.warning(f'{k} is hard coded and will not be changed')
        else:
            kw[k.upper()] = v
    kw['CATALOG_TYPE'] = 'FITS_LDAC'
    kw['extra_params'] = 'XWIN_IMAGE,YWIN_IMAGE,ERRAWIN_IMAGE,ERRBWIN_IMAGE,'
    kw['extra_params'] += 'FLUXERR_AUTO,FLAGS,FLUX_RADIUS,FWHM_IMAGE,'
    kw['extra_params'] += 'ELLIPTICITY'
    sextractor.run(image_path, cat_name, tmp_path=tmp_path, 
                   run_label=run_label, **kw)

    # update config options
    final_options = default_options.copy()
    for k, v in scamp_options.items():
        k = k.upper()
        if k not in all_option_names:
            msg = f'{k} is not a valid SCAMP option -> we will ignore it!'
            logger.warning(msg)
        else:
            logger.debug('SCAMP config update: {} = {}'.format(k, v))
            final_options[k] = v

    # build the command
    cmd = f'{executable} -c {config_path}'
    for k, v in final_options.items():
        cmd += f' -{k.upper()} {v}'
    cmd += f' {cat_name}'

    logger.debug(f'>>> {cmd}')
    call(cmd, shell=True)

    if created_tmp:
        logger.debug('Deleting temporary file ' + image_path)
        os.remove(image_path)
    if not save_cat:
        os.remove(cat_name)

    header_fn = cat_name.replace('.cat', '.head')
    scamp_header = fits.Header.fromtextfile(header_fn)

    if not save_scamp_header:
        os.remove(header_fn)
        header_fn = None

    results = DFStruct(header=scamp_header, header_path=header_fn)

    return results
