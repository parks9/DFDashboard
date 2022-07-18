import sys, os
import glob, time
import numpy as np
import pandas as pd

from ..database import DFDatabaseHub
from .. import pipelines as pipe
from .. import utils, logger


@utils.func_timer_info
def reset_databases_on_date(args):

    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)

    db_hub = DFDatabaseHub(config['db_path'],
                           config['data_path'],
                           config['mcals_path'])

    dates = utils.parse_strings_or_file_to_list(args.dates, args.dates_fn)
    logger.info('Reseting databases on the following: ' + ', '.join(dates))

    for date in dates:
        logger.info('*_____| ' + date + ' |_____*')


        index =  db_hub.frames.query_index(date=date)
        na_list = [pd.NA] * len(index)
        nullable_int_arr = pd.array(na_list, dtype='Int64')
        nullable_str_arr = pd.array(na_list, dtype='string')

        if len(na_list) > 0:
            db_hub.frames.update(
                index, 
                is_good=na_list,
                flags=nullable_int_arr,
                master_dark_id=nullable_int_arr,
                master_dark_date=nullable_str_arr,
                master_flat_id=nullable_int_arr,
                master_flat_date=nullable_str_arr,
            )

        index =  db_hub.mcals.query_index(date=date)

        if len(index) > 0:
            db_hub.mcals.update(index, exists=[False] * len(index))
                                        
    db_hub.frames.write_database(overwrite=True)
    db_hub.mcals.write_database(overwrite=True)


def main():
    args = utils.default_script_args()
    reset_databases_on_date(args)
