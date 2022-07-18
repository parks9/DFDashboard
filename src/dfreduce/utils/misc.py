"""
Functions for doing random (but useful!) things.
"""
# Standard library
import logging
import datetime
from time import time
from functools import wraps

# Third-party
import numpy as np
import pandas as pd
from astropy import units as u

# Project
from .dfio import load_path_or_header
from .. import logger


__all__ = [
    'check_astropy_units',
    'command_line_override', 
    'divide_list_into_chunks',
    'fetch_default_db_hub',
    'func_timer_debug', 
    'func_timer_info', 
    'get_image_shape', 
    'is_a_date',
    'is_list_like', 
    'log_header_label',
    'list_of_strings',
    'make_list_like_if_needed',
    'parse_to_list', 
    'parse_strings_or_file_to_list', 
    'parse_dates_to_list', 
    'reverse_dict',
    'setup_logger', 
    'slice_image_center',
    'timer'
]


def check_astropy_units(value, default_unit):
    t = type(default_unit)
    if type(value) == u.Quantity:
        quantity = value
    elif (t == u.IrreducibleUnit) or (t == u.Unit):
        quantity = value * default_unit
    elif t == str:
        quantity = value * getattr(u, default_unit) 
    else:
        raise Exception('default_unit must be an astropy unit or string')
    return quantity


def command_line_override(args, config):
    """
    Overwrite settings in config dictionary that were passed as
    command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments parsed by argparse.
    config : dict
        Pipeline configuration.

    Returns
    -------
    config :  dict
        Updated pipeline configuration.

    Notes
    -----
    The configuration dictionary is modified in place. Set the default
    command-line argument for a parameter to None if you want to use the value
    in the configuration dictionary.
    """
    for k, v in args._get_kwargs():
        if k in config.keys() and v is not None:
            config[k] = v
    return config


def divide_list_into_chunks(l, n):
    """
    Divide a list into chucks for iterating.
    """
    def _divide(l, n):
        for i in range(0, len(l), n):  
            yield l[i:i + n] 
    return list(_divide(l, n))


def func_timer_debug(f):
    """
    A function decorator to time how long it takes to execute. The time is
    printed as DEBUG using the logger.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        dt = end - start
        if dt > 120:
            dt /= 60
            unit = 'min'
        else:
            unit = 'sec'
        logger.debug('{} completed in {:.2f} {}'.format(f.__name__, dt, unit))
        return result
    return wrapper


def fetch_default_db_hub():
    from .. import DFDatabaseHub
    from .. import db_path_default, data_path_default, mcals_path_default
    db_hub = DFDatabaseHub(db_path_default, 
                           data_path_default, 
                           mcals_path_default)
    return db_hub


def func_timer_info(f):
    """
    A function decorator to time how long it takes to execute. The time is
    printed as DEBUG using the logger.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        dt = end - start
        if dt > 120:
            dt /= 60
            unit = 'min'
        else:
            unit = 'sec'
        logger.info('{} completed in {:.2f} {}'.format(f.__name__, dt, unit))
        return result
    return wrapper


def get_image_shape(path_or_header):
    header = load_path_or_header(path_or_header)
    shape = header['NAXIS2'], header['NAXIS1']
    return shape


def is_a_date(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def is_list_like(check):
    t = type(check)
    c = t == list or t == np.ndarray or t == pd.Series or\
        t == pd.Int64Index or t == pd.arrays.IntegerArray or\
        t == pd.arrays.StringArray or t == tuple
    return c


def log_header_label(path_or_header):
    header = load_path_or_header(path_or_header)
    target = header['TARGET']
    serialno = header['SERIALNO']
    date  = header['DATE'].split('T')[0]
    return f'target = {target} | serialno = {serialno} | header date = {date}'


def list_of_strings(str_or_list):
    """
    Return a list of strings from a single string of comma-separated values.

    Parameters
    ----------
    str_or_list : str or list-like
        Single string of comma-separated values or a list of strings. If it's
        the latter, then the inpits list is simply returned.

    Examples
    --------

            INPUT                                 OUTPUT
    'flag_1,flag_2,flag_3'         --> ['flag_1', 'flag_2', 'flag_3']
    'flag_1, flag_2, flag_3'       --> ['flag_1', 'flag_2', 'flag_3']
    ['flag_1', 'flag_2', 'flag_3'] --> ['flag_1', 'flag_2', 'flag_3']
    """
    if is_list_like(str_or_list):
        ls_str = str_or_list
    elif type(str_or_list) == str:
        ls_str = str_or_list.replace(' ', '').split(',')
    else:
        Exception('{} is not correct type for list of str'.format(str_or_list))
    return ls_str


def make_list_like_if_needed(obj):
    if is_list_like(obj):
        list_like_obj = obj
    else:
        list_like_obj = [obj]
    return list_like_obj


def parse_to_list(var):
    if not is_list_like(var):
        var = [var]
    return var


def parse_strings_or_file_to_list(vars=None, vars_fn=None):
    if vars_fn is not None:
        vars =  np.loadtxt(dates_fn, dtype=str)
    vars = parse_to_list(vars)
    return vars


def parse_dates_to_list(dates=None, dates_fn=None):
    return parse_strings_or_file_to_list(dates, dates_fn)


def reverse_dict(d):
    return dict(map(reversed, d.items()))


def setup_logger(level, log_fn=None):
    """
    Setup the pipeline logger.

    Parameters
    ----------
    level : str
        The log level (debug, info, warn, or error).
    log_fn : str (optional)
       Log file name.
    """
    if log_fn is not None:
        fh = logging.FileHandler(log_fn)
        msg = '[%(filename)s:%(lineno)d] %(asctime)s | '
        msg += '%(levelname)s: %(message)s'
        formatter = logging.Formatter(msg, '%Y-%m-%d | %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.setLevel(level.upper())


def slice_image_center(shape):
    """
    Generate the 2D slice for the center of an image accounting for whether 
    its dimensions are even or odd.
    """
    nrow, ncol = shape

    if nrow % 2 == 0:
        row_c = [nrow//2-1, nrow//2]
    else:
        row_c = nrow // 2
        
    if ncol % 2 == 0:
        col_c = [ncol//2-1, ncol//2]
    else:
        col_c = ncol // 2

    row_c, col_c = np.meshgrid(row_c, col_c)
    row_c = row_c.flatten()
    col_c = col_c.flatten()

    slice_c = np.s_[row_c, col_c]

    return slice_c


def timer(start_time=None, task_name='Task'):
    if start_time is None:
        current_time = time()
        return current_time
    else:
        end_time = time()
        dt = (end_time - start_time) / 3600
        unit = 'hr'
        if dt < 1:
            dt *= 60
            unit = 'min'
        logger.info(f'{task_name} completed in {dt:.2f} {unit}.')
