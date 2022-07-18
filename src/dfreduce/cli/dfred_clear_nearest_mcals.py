import sys, os
import glob, time
import numpy as np
import pandas as pd

from .. import pipelines as pipe
from ..database import DFDatabaseHub
from .. import utils, logger


@utils.func_timer_info
def clear_nearest_mcals(args, dates=None, which='both'):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)

    db_hub = DFDatabaseHub(config['db_path'],
                           config['data_path'],
                           config['mcals_path'])

    if dates is not None:
        kw = dict(dates=dates, dates_fn=None)
    else:
        kw = dict(dates=args.dates, dates_fn=args.dates_fn)
    dates = utils.parse_dates_to_list(**kw)

    lab = 'mcals' if which == 'both' else which
    logger.info(f'Clearing nearest {lab} on ' + ', '.join(dates))

    for date in dates:
        logger.info('Working on ' + date)

        index =  db_hub.frames.query_index(date=date)

        na_list = [pd.NA] * len(index)
        nullable_int_arr = pd.array(na_list, dtype='Int64')

        if len(na_list) > 0:
            if which == 'both' or which == 'darks' or which == 'dark':
                db_hub.frames.update(index, 
                                     master_dark_id=nullable_int_arr,
                                     master_dark_date=na_list)
            if which == 'both' or which == 'flats' or which == 'flat':
                db_hub.frames.update(index, 
                                     master_flat_id=nullable_int_arr,
                                     master_flat_date=na_list)
                                        
    db_hub.frames.write_database(overwrite=True)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='classify dark-subtracted flats')

    # required arguments
    parser.add_argument('-c', '--config-fn', required=True,
                        help='flats configuration filename')

    # optional arguments
    parser.add_argument('--log-level', default='info',
                        help='log level (debug, info, warn, error)')
    parser.add_argument('--log-fn', default=None, help='output log file name')
    parser.add_argument('--which', default='both', help='both, darks, or flats')

    # mutually exclusive arguments (one is required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dates', nargs='*')
    group.add_argument('--dates-fn', help='name of file with list of dates')

    args = parser.parse_args()
    utils.setup_logger(args.log_level, args.log_fn)

    clear_nearest_mcals(args, which=args.which)
