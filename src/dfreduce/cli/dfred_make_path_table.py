# Standard library
from argparse import ArgumentParser

# Project
from .. import utils
from ..database import DFDatabaseHub


def make_path_table(args):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)
    db_hub = DFDatabaseHub(config['db_path'],
                           config['data_path'],
                           config['mcals_path'])
    path_table = db_hub.light_reduction_path_table(args.target)
    return path_table


def main():
    parser = ArgumentParser()

    # required arguments
    parser.add_argument('-c', '--config-fn', 
                        help='configuration file', required=True)
    parser.add_argument('-t', '--target', required=True, help='Target field')
    parser.add_argument('-o', '--out-fn', required=True, help='csv file name')
    args = parser.parse_args()

    path_table = make_path_table(args)
    path_table.to_csv(args.out_fn)
