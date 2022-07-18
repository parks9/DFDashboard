import os
from subprocess import call
from astropy.io import fits
from ..detection import sextractor
from .. import DFStruct
from .. import package_dir
from .. import utils, logger


input_file_path = os.path.join(package_dir, 'improc/swarp_files')
default_config = os.path.join(input_file_path, 'default.config')


# get list of all config options
file = open(os.path.join(input_file_path, 'config_options.txt'), 'r')
all_option_names = [line.rstrip() for line in file]
file.close()


default_copy_kw = 'TARGET,FILTNAM,FILTER,SERIALNO,ALTITUDE,AZIMUTH,ZP,ZPRMS,'
default_copy_kw += 'ZPNGOOD,ZPNREJ,FWHM,SSIGMA,AXRATIO,AXRRMS,THMEAN,THRMS,'
default_copy_kw += 'SKYLVL,SKYRMS,SKYSB,DATE,MEDSKY,IMAGETYP,ADECSYS,OBJCTRA,'
default_copy_kw += 'OBJCTDEC,NOBJ,MEDELLIP,MEDFWHM'


# default non-standard options
default_options = dict(
    SUBTRACT_BACK='N',
    BACK_SIZE=256,
    GAIN_DEFAULT=0.37,
    CELESTIAL_TYPE='EQUATORIAL',
    RESAMPLING_TYPE='LANCZOS3',
    PIXELSCALE_TYPE='MANUAL',
    PIXEL_SCALE=2.5,
    FSCALASTRO_TYPE='VARIABLE',
    FSCALE_KEYWORD='IGNOREME',
    IMAGE_SIZE='5750,4250',
    CENTER_TYPE='MANUAL',
    WRITE_XML='N',
)


def _convert_coord_string(coord):
    coord = utils.to_skycoord(coord).to_string('hmsdms')
    coord = coord.replace(' ', ',').\
                  replace('h', ':').\
                  replace('m', ':').\
                  replace('d', ':').\
                  replace('s', '')
    return coord
 

def run(path_or_pixels, out_fn, center_radec, header=None, tmp_path='/tmp', 
        run_label=None, executable='swarp', config_path=default_config, 
        extra_header=None, copy_kw=default_copy_kw, save_output_file=True, 
        **swarp_options):

    image_path, created_tmp = utils.temp_fits_file(path_or_pixels,
                                                   tmp_path=tmp_path,
                                                   run_label=run_label,
                                                   prefix='swarp_tmp',
                                                   header=header)

    if extra_header is not None:
        extra_header_fn = image_path.replace('.fits', '.head')
        extra_header.totextfile(extra_header_fn, overwrite=True)

    # update config options
    final_options = default_options.copy()
    for k, v in swarp_options.items():
        k = k.upper()
        if k not in all_option_names:
            msg = f'{k} is not a valid SWarp option -> we will ignore it!'
            logger.warning(msg)
        else:
            logger.debug('SWarp config update: {} = {}'.format(k, v))
            final_options[k] = v
    final_options['CENTER'] = _convert_coord_string(center_radec)
    final_options['IMAGEOUT_NAME'] = out_fn

    # build the command
    cmd = f'{executable} -c {config_path}'
    for k, v in final_options.items():
        cmd += f' -{k.upper()} {v}'
    cmd += f' {image_path}'

    logger.debug(f'>>> {cmd}')
    call(cmd, shell=True)
    pixels, header = fits.getdata(out_fn, header=True)

    # clean up temporary files
    if created_tmp:
        logger.debug('Deleting temporary file ' + image_path)
        os.remove(image_path)
    if extra_header is not None:
        os.remove(extra_header_fn)
    if not save_output_file:
        os.remove(out_fn)
        out_fn = None

    results = DFStruct(pixels=pixels, header=header, path=out_fn)

    return results
