import os
import shutil
import numpy as np
import pandas as pd

from ..log import logger
from ..database import DFDatabaseHub
from .. import tasks
from .. import utils


@utils.func_timer_info
def create_master_flats(args):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)
    dates = utils.parse_strings_or_file_to_list(args.dates, args.dates_fn)

    db_hub = DFDatabaseHub(config['db_path'], 
                           config['data_path'], 
                           config['mcals_path'])

    greeting = 'Creating master flats on the following dates: {}'
    logger.info(greeting.format(', '.join(dates)))
    mflat_kw = config.pop('create_master_flat', {})

    for date in dates:
        logger.info('Creating master flats for ' + date)
        kw = dict(frame_type='flat', date=date, is_good=True)
        good_flats = db_hub.frames.query('serialno', **kw)

        out_dir = os.path.join(db_hub.mcals.flat_path, date)
        if os.path.isdir(out_dir):
            if len(os.listdir(out_dir)) > 0:
                logger.warning(f'Deleting previous master flats from {date}')
                idx = db_hub.mcals.query_index(frame_type='flat', date=date)
                if len(idx) > 0:
                    db_hub.mcals.update(idx, exists=[False] * len(idx))
                shutil.rmtree(out_dir)
        utils.mkdir_if_needed(out_dir)

        group_sn = good_flats.groupby('serialno')
        for sn, idx in group_sn.groups.items():
            logger.info('Working on {} '.format(sn))
            mflat = tasks.flats.create_master_flat(idx, db_hub, **mflat_kw)
            if mflat is None:
                logger.critical('No master flat for {} on {}'.format(sn, date))
            else:
                fn = db_hub.mcals.build_flat_path(date, sn)
                utils.write_pixels(fn, mflat.pixels, mflat.header)
        db_hub.mcals.add_night(date, 'flats')
        db_hub.mcals.write_database(overwrite=True)
        logger.info('Finished creating master flats for {}'.format(date))


def main():
    args = utils.default_script_args()
    create_master_flats(args)
