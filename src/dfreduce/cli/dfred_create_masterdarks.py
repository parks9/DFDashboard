# Standarc libraru
import os
import shutil

# Third-party
import numpy as np
from  tqdm import tqdm

# Project
from ..database import DFIndividualFrames, DFMasterCals
from .. import utils, tasks, logger


@utils.func_timer_info
def create_master_darks(args):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)

    dates = utils.parse_strings_or_file_to_list(args.dates, args.dates_fn)

    greeting = 'Creating master darks on the following dates: {}'
    logger.info(greeting.format(', '.join(dates)))

    # load the databases
    frames_db = DFIndividualFrames(config['db_path'], config['data_path'])
    mcals_db = DFMasterCals(config['db_path'], config['mcals_path'])
    min_num_darks = config['create_master_dark'].pop('min_num_darks', 2)


    for date in dates:
        logger.info('Creating master darks for {}...'.format(date))

        # grab darks that have been classified as "good"
        results = frames_db.query(select='serialno,exptime', 
                                  frame_type='dark', 
                                  date=date, 
                                  is_good=True)

        # create output directory if necessary
        out_dir = os.path.join(mcals_db.dark_path, date)
        if os.path.isdir(out_dir):
            if len(os.listdir(out_dir)) > 0:
                logger.warning(f'Deleting previous master darks from {date}')
                idx = mcals_db.query_index(frame_type='dark', date=date)
                if len(idx) > 0:
                    mcals_db.update(idx, exists=[False] * len(idx))
                shutil.rmtree(out_dir)
        utils.mkdir_if_needed(out_dir)

        # the following can maybe be a pipeline? 
        # make a master dark for each exposure time / serialno combo
        logger.start_tqdm()
        for group in tqdm(results.groupby(['serialno', 'exptime']).groups):
            sn, t = group
            logger.debug('serialno: {}   exptime: {}'.format(sn, t))

            # Select "good" darks with this serialno/exptime combo
            dark_paths = frames_db.query_frame_paths(
                frame_type='dark', date=date, is_good=True,
                serialno=sn, exptime=t, full_paths=True)

            # Create the master dark
            if len(dark_paths) >= min_num_darks:
                tasks.darks.create_master_dark(dark_paths.values,
                                               save_path=out_dir, serialno=sn)
            else:
                msg = 'Not enough good darks to make master | '
                msg += f'{date} | {sn} | {t} | N_dark = '
                logger.critical(msg + f'{len(dark_paths)} < {min_num_darks}')

        logger.end_tqdm()
        mcals_db.add_night(date, 'darks')

        logger.info(f'Writing {date} master darks to database.')
        mcals_db.write_database(overwrite=True)


def main():
    args = utils.default_script_args()
    create_master_darks(args)
