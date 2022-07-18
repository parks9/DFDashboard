"""write_every=False,
Module for common command-line arguments.
"""
# Standard library
import os
import sys
from argparse import ArgumentParser

# Project
from .. import utils, logger


__all__ = ['default_script_parser', 'default_script_args']


def default_script_parser(itervar='date', write_every=False, nseg=False, 
                          target_coord_group=False, extra_args={}):
    parser = ArgumentParser()

    # mutually exclusive arguments (one is required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(f'--{itervar}s', nargs='*')
    group.add_argument(f'--{itervar}', dest=f'{itervar}s', type=str)
    group.add_argument(f'--{itervar}s-fn', 
                       help=f'name of file with list of {itervar}s')

    # required if default config environment not defined
    default_fn = os.getenv('DFRED_DEFAULT_CONFIG')
    required = True if default_fn is None else False
    parser.add_argument('-c', '--config-fn', default=default_fn,
                        help='configuration file', required=required)

    # optional arguments
    parser.add_argument('--log-level', default='info',
                        help='log level (debug, info, warn, error)')
    parser.add_argument('--log-fn', default=None, help='output log file name')
    parser.add_argument('--nproc', default=1, type=int)

    # arguments only needed by some scripts 
    if write_every:
        parser.add_argument('--write-every', default=50, type=int,
                            help='Frequency to write database updates.')
    if nseg:
        parser.add_argument('--nseg', default=10,
                            help='stack image in nseg x nseg grid')
    if target_coord_group:
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--survey', default='UW')
        group.add_argument('--coord', dest='coords', type=str)
        group.add_argument('--coords', nargs='*')
        group.add_argument('--coords-fn')


    # add extra arguments if needed
    for argument, kw in extra_args.items():
        parser.add_argument(f'--{argument}', **kw)

    return parser


def default_script_args(itervar='date', **kwargs):
    parser = default_script_parser(itervar, **kwargs)
    args = parser.parse_args()
    utils.setup_logger(args.log_level, args.log_fn)

    if args.log_fn is not None:
        import pprint
        cmd = ' '.join(sys.argv)
        logger.info(f"*****************{'*' * len(cmd)}")
        logger.info(f'Running Command: {cmd}')
        logger.info(f"*****************{'*' * len(cmd)}")
        with open(args.log_fn, 'a') as f:
            config = utils.load_config(args.config_fn)
            print('------------------------', file=f)
            print('START CONFIGURATION FILE', file=f)
            print('------------------------', file=f)
            pprint.pprint(config, f)
            print('-----------------------', file=f)
            print('END CONFIGURATION FILE', file=f)
            print('----------------------', file=f)

    return args
