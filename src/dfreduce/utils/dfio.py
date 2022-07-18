"""
Functions for reading input and writing output.
"""
# Standard library
import os
import shutil
try:
    import cPickle as pickle
except:
    import pickle

# Third-party
import yaml
import numpy as np
from astropy.io import fits

# Project
from .. import logger, DFStruct
from .. import data_path_default, db_path_default, mcals_path_default


__all__ = [
    'default_header_history',
    'is_file_corrupt',
    'mkdir_if_needed',
    'load_config',
    'load_path_or_header',
    'load_path_or_pixels',
    'load_pixels_and_header',
    'load_pickled_data',
    'pickle_data',
    'print_config',
    'update_header',
    'temp_fits_file',
    'write_pixels',
]


def default_header_history(prefix=None):
    """
    Return default header line with the date and user name.
    """
    import getpass
    from .coordinates import get_today
    if prefix == None:
        msg = 'Created/updated with DFReduce by {} on {}'
    else:
        msg = prefix + ' with DFReduce by {} on {}'
    return msg.format(getpass.getuser(), get_today())


def is_file_corrupt(path_or_pixels):
    if not (type(path_or_pixels) == str or type(path_or_pixels) == np.str_):
        header = None
        pixels = path_or_pixels
        is_corrupt = False
    else:
        try:
            pixels, header = load_pixels_and_header(path_or_pixels)
            is_corrupt = False
        except:
            if not os.path.isfile(path_or_pixels):
                raise Exception(f'{path_or_pixels} does not exist')
            is_corrupt = True
            pixels = None
            header = None
    results = DFStruct(pixels=pixels, header=header, is_corrupt=is_corrupt)
    return results 


def load_config(config_fn):
    """
    Load  a yaml configuration file into a dictionary.

    Parameters
    ----------
    config_fn : str
        Configuration file name.

    Returns
    -------
    config : dict
        The configuration.
    """
    with open(config_fn, 'r') as fn:
        logger.debug('Loading configuration file: ' + config_fn)
        config = yaml.load(fn, Loader=yaml.FullLoader)
    msg = 'not given --> using default:'
    if 'db_path' not in config.keys():
        logger.info(f'db_path {msg} {db_path_default}')
        config['db_path'] = db_path_default
    if 'mcals_path' not in config.keys():
        logger.info(f'mcals_path {msg} {mcals_path_default}')
        config['mcals_path'] = mcals_path_default
    if 'data_path' not in config.keys():
        logger.info(f'data_path {msg} {data_path_default}')
        config['data_path'] = data_path_default
    config['paths'] = dict(
        db_path = config['db_path'],
        data_path = config['data_path'],
        mcals_path = config['mcals_path'],
    )
    return config


def load_path_or_header(path_or_header):
    _type = type(path_or_header)
    if _type == str or _type == np.str_:
        header = fits.getheader(path_or_header)
    elif _type == fits.Header:
        header = path_or_header
    else:
        logger.critical('{} is not a valid path or header type!'.format(_type))
        header = None
    return header


def load_path_or_pixels(path_or_pixels, dtype=None, bounds=None):
    """
    Check if the input is a file or numpy array and return a numpy array.

    Parameters
    ----------
    image_path_or_pixels : str or ndarray or list of one of these
        An image file name or numpy array of its pixels.
    dtype : type (optional)
        If not None, change the image to this data type.

    Returns
    -------
    image_pixels : ndarray
        The image pixels.
    bounds : array (optional)
        Array containing pixel values to trim data in format:
        (x_start,x_end,y_start,y_end).

    Notes
    -----
    This function is useful for making it possible to pass either an image
    file name or the pixels.
    """
    data = path_or_pixels
    if type(data) == str or type(data) == np.str_:
        data = fits.getdata(data)
        if bounds is not None:
            if data.shape != (bounds[1]-bounds[0],bounds[3]-bounds[2]):
                data = data[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    if dtype is not None:
        data = data.astype(dtype)
    return data


def load_pixels_and_header(path_or_pixels, path_or_header=None):
    if not (type(path_or_pixels) == str or type(path_or_pixels) == np.str_):
        msg = 'Must provide header if pixels are given!'
        assert path_or_header is not None, msg
        header = load_path_or_header(path_or_header)
    else:
        if path_or_header is not None:
            header = load_path_or_header(path_or_header)
        else:
            header = load_path_or_header(path_or_pixels)
    pixels = load_path_or_pixels(path_or_pixels)
    return pixels, header


def load_pickled_data(filename):
    """
    Load pickled data

    input
    -----
    filename : string. name of file that
        contains pickled data

    output
    ------
    the unpickled data
    """
    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def mkdir_if_needed(directory, force_empty=False):
    """"
    Create directory if it does not exist.
    """
    if not os.path.isdir(directory):
        logger.info('Creating directory: ' + directory)
        os.makedirs(directory)
    elif force_empty:
        if len(os.listdir(directory)) > 0:
            logger.warning(f'Deleting previous content of {directory}')
            shutil.rmtree(directory)
            os.makedirs(directory)


def pickle_data(filename, data, warn_overwrite=True):
    """
    Pickle data

    input
    -----
    filename : string. name of file to
        output with full path name.
    data : any data construct. the data
        to be pickled.
    """
    if os.path.isfile(filename) and warn_overwrite:
        logger.warning(filename + ' exists. Will overwrite.')
    pkl_file = open(filename, 'wb')
    logger.debug('Pickling data to ' + filename)
    pickle.dump(data, pkl_file)
    pkl_file.close()
    
    
def print_config(config):
    import pprint
    pp = pprint.PrettyPrinter()
    if type(config) == str:
        config = load_config(config)
    pp.pprint(config)


def temp_fits_file(path_or_pixels, tmp_path='/tmp', run_label=None, 
                   prefix='tmp',  header=None):
    is_str = type(path_or_pixels) == str or type(path_or_pixels) == np.str_
    if is_str and header is None:
        path = path_or_pixels
        created_tmp = False
    else:
        if is_str:
            path_or_pixels = fits.getdata(path_or_pixels)
        label = '' if run_label is None else '_' + run_label
        fn = '{}{}.fits'.format(prefix, label)
        path = os.path.join(tmp_path, fn)
        logger.debug('Writing temporary fits file {}'.format(path))
        fits.writeto(path, path_or_pixels, header=header, overwrite=True)
        created_tmp = True
    return path, created_tmp


def update_header(path_or_header, msg=None, **kwargs):
    header = load_path_or_header(path_or_header)
    for k, v in kwargs.items():
        header[k] = v
    header.add_history(default_header_history(msg))
    return header


def write_pixels(file_name, pixels, header=None):
    """
    Write pixels to fits file.

    Parameters
    ----------
    file_name : str
        The file name.
    pixels : ndarray
        Image pixels.
    header : astropy.io.fits.Header (optional)
        Image header.

    """
    if os.path.isfile(file_name):
        logger.warning(file_name + ' exists -- will overwrite')
    logger.debug('Writing ' + file_name)
    fits.writeto(file_name, pixels, header=header, overwrite=True)
