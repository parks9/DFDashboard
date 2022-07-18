# Standard library
import os
from argparse import ArgumentParser

# Project
from .. import utils, logger, improc
from ..database import DFDatabaseHub
from ..pipelines import lights as pipe


class Worker(object):

    def __init__(
        self, 
        db_hub, 
        config, 
        path_table, 
        target_radec, 
        ref_cat,
        out_path,
        sip_head_path,
        coadd,
        write_every
    ):
        self.db_hub = db_hub
        self.config = config
        self.path_table = path_table
        self.num_frames = len(path_table)
        self.target_radec = target_radec
        self.out_path = out_path
        self.sip_head_path = sip_head_path
        self.ref_cat = ref_cat
        self.coadd = coadd
        self.num_complete = 0
        self.write_every = write_every

    def work(self, frame_id):
        results = pipe.process_light_deep(
            target_radec=self.target_radec, 
            light_path=self.path_table.loc[frame_id, 'light'],
            dark_path=self.path_table.loc[frame_id, 'master_dark'],
            flat_path=self.path_table.loc[frame_id, 'master_flat'],
            ref_cat=self.ref_cat,
            out_path=self.out_path,
            sip_head_path=self.sip_head_path,
            deep_coadd_pixels=self.coadd.pixels,
            deep_coadd_header=self.coadd.header,
            run_label=f'light_{frame_id}',
            config=self.config
        )
        return frame_id, results['flags'].int_val

    def callback(self, results):
        self.num_complete += 1
        frame_id, flag_val = results
        is_good = flag_val == 0
        self.db_hub.frames.update(frame_id, flags=flag_val, is_good=is_good)
        mod_num_complete = self.num_complete % self.write_every
        if mod_num_complete == 0 or self.num_complete == self.num_frames:
            logger.debug(f'Writing last {self.write_every} database updates.')
            frac_complete = self.num_complete / self.num_frames
            self.db_hub.frames.write_database(overwrite=True)


@utils.func_timer_info
def process_lights_deep(args, targets, write_every=None):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)
    db_hub = DFDatabaseHub(**config['paths'])
    targets = utils.parse_to_list(targets)

    if args.coords is not None or args.coords_fn is not None:
        coords = utils.parse_strings_or_file_to_list(args.coords,
                                                     args.coords_fn)
        skycoords = utils.to_skycoord_list(coords)
    elif args.survey is not None:
        skycoords = []
        radec_fetcher = utils.survey_radec_fetcher[args.survey]
        for target in targets:
            skycoords.append(radec_fetcher(target))
    else:
        raise Exception('How we supposed to find your target(s)?')

    for target, skycoord in zip(targets, skycoords):
        # build path table for target
        path_table = db_hub.light_reduction_path_table(target)

        if path_table is None:
            logger.critical(f'Zero lights found for field {target}')
            continue

        frame_ids = path_table.index

        logger.info(f'Processing {len(path_table)} lights for field {target}.')

        # check paths
        target_path = os.path.join(config['reduced_path'], target)
        if not os.path.isdir(target_path):
            msg = f'Target reduction path does not exist for field {target}. '
            logger.critical(msg + 'Have you run process_light for this frame?')
            continue
        coadd_path = os.path.join(target_path, config['coadd_dir'])
        if not os.path.isdir(target_path):
            msg = f'Target coadd path does not exist for field {target}. '
            logger.critical(msg + 'Have you run process_light for this frame?')
            continue
        sip_head_path = os.path.join(target_path, config['sip_header_dir'])
        if not os.path.isdir(sip_head_path):
            msg = f'SIP Header directory for {target} not found. '
            logger.warning(msg + 'Will need to rerun astrometry.')
            sip_head_path = None

        # make deep coadd image
        logger.info(f'Making deep median coadd for field {target}')
        coadd = improc.deep_stack(coadd_path)
        if coadd.pixels is None:
            logger.critical(f'No median coadds for field {target} exist.')
            continue

        # make paths if necessary
        deep_path = os.path.join(target_path, config['deep_dir'])
        utils.mkdir_if_needed(deep_path, force_empty=True)
        deep_reg_path = os.path.join(deep_path, config['registered_dir'])
        utils.mkdir_if_needed(deep_reg_path)

        # load reference catalog for photometry
        corners = utils.get_fov_corners(skycoord, scale=[1.5, 1.7])
        ref_cat = utils.load_apass_in_region(
            config['ref_cat_path'], bounds=corners.bounds)

        if write_every is None:
            write_every = len(path_table)

        # start working using nproc processes
        name = 'PROCESS LIGHTS DEEP'
        worker = Worker(
            db_hub, 
            config, 
            path_table, 
            skycoord, 
            ref_cat, 
            deep_reg_path,
            sip_head_path,
            coadd,
            write_every
        )
        utils.multiproc.work_it(worker, frame_ids, args.nproc, name)

        # switch back to single processor mode and check zero points 
        path_table = db_hub.light_reduction_path_table(target)
        results = pipe.reject_aberrant_zero_points(
            path_table=path_table, 
            reg_out_path=deep_reg_path,
            max_zp_offset=config['max_zp_offset'], 
            target=target,
            suffix='reg_deep'
        )

        # update database
        frame_id = results.flag_arr.index.values
        flags = results.flag_arr.to_integer()['flags'].values
        is_good = results.flag_arr.is_good().values
        db_hub.frames.update(frame_id, flags=flags, is_good=is_good)
        logger.info(f'Writing zero point flags to database.')
        db_hub.frames.write_database(overwrite=True)


def main():
    args = utils.default_script_args('target',
                                     write_every=True,
                                     target_coord_group=True)
    process_lights_deep(args, args.targets, args.write_every)
