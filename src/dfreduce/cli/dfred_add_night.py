# Standard library
import os

# Project
from ..database import DFIndividualFrames
from .. import utils, logger


@utils.func_timer_info
def add_night(args, df_unit=None):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)
    dates = utils.parse_strings_or_file_to_list(args.dates, args.dates_fn)

    db = DFIndividualFrames(config['db_path'], config['data_path'])

    logger.info('Adding night(s) to database: {}'.format(', '.join(dates)))

    for date in dates:
        msg = f'Adding data from {date}'
        if df_unit is not None:
            msg += f' for Dragonfly{df_unit}'
        logger.info(f'----- {msg} -----')
        db.add_night(date, df_unit=df_unit)

    logger.info('Writing all updates to csv file')
    db.write_database(overwrite=True)
    logger.info('Finished adding data to database.')


def main():
    parser = utils.default_script_parser()
    parser.add_argument('--df-unit', default=None, type=int)
    args = parser.parse_args()
    utils.setup_logger(args.log_level, args.log_fn)
    add_night(args, df_unit=args.df_unit)
