# Standard library
import os
from glob import glob
from argparse import ArgumentParser

# Project
from .. import utils, logger
from ..database import DFDatabaseHub
from ..improc import MedianCoadder, WeightedAverageCoadder


@utils.func_timer_info
def stack_target(args, targets, nseg, coadd_type, deep):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)
    targets = utils.parse_to_list(targets)

    for target in targets:
        reduce_path =  os.path.join(config['reduced_path'], target)
        if deep:
            logger.info(f'Making DEEP coadd for field {target}.')
            reduce_path =  os.path.join(reduce_path, config['deep_dir'])
        reg_out_path = os.path.join(reduce_path, config['registered_dir'])
        if not os.path.isdir(reg_out_path):
            logger.critical(f'Registered path {reg_out_path} does not exist.')
            continue
        suffix = 'reg' if not deep else 'reg_deep'
        paths = glob(os.path.join(reg_out_path, f'*_{suffix}.fits'))
        if len(paths) <= 1:
            logger.critical(f'Not enough frames to stack field {target}.')
            continue
        logger.info(f'Found {len(paths)} registered frames for {target}.')
        
        coadd_path = os.path.join(reduce_path, config['coadd_dir'])
        utils.mkdir_if_needed(coadd_path)
        bpm_path = config.pop('bpm_path', '/tmp')

        ctype = coadd_type
        suffix = '' if not deep else '_deep'

        for b in 'gr':
            med_fn = os.path.join(coadd_path, f'coadd_median_{b}{suffix}.fits')
            needs_med_coadd = not os.path.isfile(med_fn)
            if ctype == 'median' or (ctype == 'average' and needs_med_coadd):
                logger.info(f'Making {b}-band median stack for {target}.')
                logger.info(f'Stacking image in {nseg} x {nseg} grid.')
                coadder = MedianCoadder(paths)
                coadder.stack_images(
                    bandpass=b, 
                    nseg=nseg, 
                    out_path=coadd_path, 
                    suffix=suffix.replace('_', '')
                )
            if ctype == 'average':
                logger.info(f'Making {b}-band average stack for {target}.')
                logger.info(f'Stacking image in {nseg} x {nseg} grid.')
                coadder = WeightedAverageCoadder(paths)
                coadder.stack_images(
                    bandpass=b,
                    median_coadd=med_fn,
                    bpm_path=bpm_path,
                    nseg=nseg,
                    nproc=args.nproc,
                    out_path=coadd_path,
                    suffix=suffix.replace('_', '')
                )


def main():
    parser = ArgumentParser()

    # required arguments
    parser.add_argument('-c', '--config-fn',
                        help='configuration file', required=True)
    # mutually exclusive arguments (one is required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--targets', nargs='*')
    group.add_argument('--target', dest='targets', type=str)

    # optional arguments
    parser.add_argument('--nseg', default=10, 
                        help='stack image in nseg x nseg grid')
    parser.add_argument('--log-level', default='info',
                        help='log level (debug, info, warn, error)')
    parser.add_argument('--log-fn', default=None, help='output log file name')
    parser.add_argument('--nproc', default=1, 
                        help='number of processors', type=int)
    parser.add_argument('--deep', action='store_true', 
                        help='set if making coadd after deep sky subtraction')
    parser.add_argument('--coadd-type', default='median',
                        help='median or (weighted) average coadd')
    args = parser.parse_args()
    utils.setup_logger(args.log_level, args.log_fn)

    stack_target(args, args.targets, args.nseg, args.coadd_type, args.deep)
