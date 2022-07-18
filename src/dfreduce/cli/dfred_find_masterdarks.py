# Standard library
import os

# Third-party
import numpy as np
import pandas as pd
from tqdm import tqdm

# Project
from ..database import DFIndividualFrames, DFMasterCals, DFDatabaseHub
from .. import utils, tasks, logger
from .dfred_clear_nearest_mcals import clear_nearest_mcals


@utils.func_timer_info
def find_master_darks(args, dates=None):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)
    db_hub = DFDatabaseHub(**config['paths'])

    if dates is not None:
        kw = dict(dates=dates, dates_fn=None)
    else:
        kw = dict(dates=args.dates, dates_fn=args.dates_fn)
    dates = utils.parse_dates_to_list(**kw)

    for date in dates:
        logger.info('Assigning master darks to lights for ' + date)
        db_this_date = db_hub.frames.query('frame_type', date=date)

        # reset any previous flags
        logger.debug('Resetting assigned master darks for ' + date)
        NA_list = [pd.NA] * len(db_this_date.index)
        db_hub.frames.update(
            db_this_date.index, 
            master_dark_id = NA_list,
            master_dark_date = NA_list
        )

        logger.start_tqdm()
        for f_id in tqdm(db_this_date.index):
            frame_type = db_hub.frames.loc[f_id, 'frame_type']
            mdark = db_hub.find_nearest_dark(f_id)
            if mdark is not None:
                if frame_type == 'flat' and date != mdark.date:
                    msg = f'Flat {f_id} date different from master dark.'
                    logger.warning(msg + ' Will skip.')
                else:
                    db_hub.frames.update(
                        f_id, 
                        master_dark_id=mdark.frame_id,
                        master_dark_date=mdark.date
                    )
        logger.end_tqdm()

        logger.info(f'Finished assigning {date} master darks.')
        db_hub.frames.write_database(overwrite=True)


def main():
    args = utils.default_script_args()
    clear_nearest_mcals(args, args.dates, 'darks')
    find_master_darks(args)
