# Standard library
import os

# Third-party
import numpy as np
import pandas as pd
from tqdm import tqdm

# Project
from ..database import DFDatabaseHub
from .. import utils, tasks, logger
from .dfred_clear_nearest_mcals import clear_nearest_mcals


@utils.func_timer_info
def find_master_flats(args, dates=None):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)

    if dates is not None:
        kw = dict(dates=dates, dates_fn=None)
    else:
        kw = dict(dates=args.dates, dates_fn=args.dates_fn)
    dates = utils.parse_dates_to_list(**kw)
    db_hub = DFDatabaseHub(config['db_path'],
                           config['data_path'],
                           config['mcals_path'])
    for date in dates:
        logger.info('Finding master flats for ' + date)
        db_this_date = db_hub.frames.query('frame_type', date=date)

        # reset any previous flags
        logger.debug('Reseting assigned master flats for ' + date)
        NA_list = [pd.NA] * len(db_this_date.index)
        db_hub.frames.update(
            db_this_date.index,
            master_flat_id = NA_list,
            master_flat_date = NA_list
        )

        logger.start_tqdm()
        for f_id in tqdm(db_this_date.index):
            master_flat = db_hub.find_nearest_flat(f_id)
            if master_flat is not None:
                db_hub.frames.update(f_id, master_flat_id=master_flat.frame_id,
                                     master_flat_date=master_flat.date)
        logger.end_tqdm()
    db_hub.frames.write_database(overwrite=True)


def main():
    args = utils.default_script_args()
    clear_nearest_mcals(args, args.dates, 'flats')
    find_master_flats(args)
