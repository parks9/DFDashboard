# Standard library
from argparse import ArgumentParser

# Project
from .. import logger, utils, cli
from ..database import DFIndividualFrames


@utils.func_timer_info
def reassign_mcals(args):
    logger.info('***** Reassigning calibration frames *****')
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)
    db = DFIndividualFrames(config['db_path'], config['data_path'])
    dates = db.df.date.unique().tolist()

    cli.clear_nearest_mcals(args, dates, 'darks')
    cli.find_master_darks(args, dates)

    cli.clear_nearest_mcals(args, dates, 'flats')
    cli.find_master_flats(args, dates)


def main():
    parser = ArgumentParser()

    # required arguments
    parser.add_argument('-c', '--config-fn', 
                        help='configuration file', required=True)

    args = parser.parse_args()
    reassign_mcals(args)
