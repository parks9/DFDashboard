# Third-party
import numpy as np
import pandas as pd

# Project
from ..database import DFDatabaseHub
from ..flags import LightFlags
from .. import pipelines as pipe
from .. import utils, logger


class Worker(object):

    def __init__(self, db_hub, config, num_frames, write_every=50):
        self.db_hub = db_hub
        self.config = config
        self.num_frames = num_frames
        self.num_complete = 0
        self.write_every = write_every

    def work(self, frame_id):
        label = f'light_{frame_id}'
        info = self.db_hub.get_frame_mcal_info(frame_id)
        if utils.is_file_corrupt(info.frame_path).is_corrupt:
            is_good = False
            int_val = LightFlags.str_to_int['CORRUPT_FILE']
            result = frame_id, int_val, is_good
        elif info.has_mcals:
            r = pipe.lights.assess_quality(
                info.frame_path, 
                info.dark_path, 
                info.flat_path, 
                run_label=label, 
                config=self.config
            )
            is_good = r['flags'].int_val == 0
            result = frame_id, r['flags'].int_val, is_good
        else:
            flags = LightFlags()
            if info.dark_path is None:
                flags.set('NO_MASTER_DARK')
            if info.flat_path is None:
                flags.set('NO_MASTER_FLAT')
            lab_d = 'master dark' if info.dark_path is not None else None
            lab_f = 'master flat' if info.flat_path is not None else None
            which = [lab for lab in [lab_d, lab_f] if lab is not None]
            missing = ' & '.join(which)
            is_good = flags.int_val == 0
            result = frame_id, flags.int_val, is_good
            logger.debug(f'light {frame_id} is missing: ' + missing)
        return result

    def callback(self, result):
        self.num_complete += 1
        frame_id, flag_val, is_good = result
        self.db_hub.frames.update(frame_id, flags=flag_val, is_good=is_good)
        mod_num_complete = self.num_complete % self.write_every
        if mod_num_complete == 0 or self.num_complete == self.num_frames:
            frac_complete = self.num_complete / self.num_frames
            logger.debug(f'Completed {100 * frac_complete:.2f}% of frames')
            self.db_hub.frames.write_database(overwrite=True)


@utils.func_timer_info
def check_lights(args):
    config = utils.load_config(args.config_fn)
    utils.command_line_override(args, config)
    dates = utils.parse_strings_or_file_to_list(args.dates, args.dates_fn)

    db_hub = DFDatabaseHub(config['db_path'], 
                           config['data_path'], 
                           config['mcals_path'])

    date_q = [f"date=='{d}'" for d in dates]

    # pandas query breaks if the query is too long
    if len(date_q) <= 20:
        q = "frame_type=='light' & (" + ' | '.join(date_q) + ')'
        frame_ids = db_hub.frames.query_index(q)
    else:
        frame_ids = []
        for d_q in  utils.divide_list_into_chunks(date_q, 20):
            q = "frame_type=='light' & (" + ' | '.join(d_q) + ')'
            frame_ids.append(db_hub.frames.query_index(q).values)
        frame_ids = np.hstack(frame_ids)
    num_frames = len(frame_ids)

    # reset flags if there are any
    db_hub.frames.loc[frame_ids, 'flags'] = pd.NA

    logger.info(f"Checking {num_frames} lights from dates: {', '.join(dates)}")
    worker = Worker(db_hub, config, num_frames)
    utils.multiproc.work_it(worker, frame_ids, args.nproc, 'CHECK LIGHTS')


def main():
    args = utils.default_script_args()
    check_lights(args)
