import sys, os
import glob, time
import numpy as np
import pandas as pd

from .. import pipelines as pipe
from ..driver import Driver
from .. import utils, logger


@utils.func_timer_info
def check_flats(args, checkpoint_path=None):

    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            driver = Driver.from_pickle(checkpoint_path)
    else:
        driver = Driver(config)

    dates = utils.parse_strings_or_file_to_list(args.dates, args.dates_fn)
    logger.info('Classifying flats taken on {}'.format(', '.join(dates)))

    for date in dates:
        logger.info('Working on ' + date)
        pipe.classify_flats(date, driver)
        driver.db_hub.frames.update(driver.trunk.index, 
                                    flags=driver.trunk['flags'],
                                    is_good=driver.trunk.is_good)
        driver.db_hub.frames.write_database(overwrite=True)
        driver.delete_trunk()
        driver.completed_steps = []


def main():
    parser = utils.default_script_parser()
    parser.add_argument('--checkpoint-path', default=None, type=str)
    args = parser.parse_args()
    check_flats(args, checkpoint_path=args.checkpoint_path)
