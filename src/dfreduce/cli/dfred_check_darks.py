# Standard library
from argparse import ArgumentParser
from multiprocessing import Pool

# Third-party
import numpy as np

# Project
from .. import utils, logger
from .. import pipelines as pipe
from ..database import DFIndividualFrames


class Worker(object):

    def __init__(self, db, config, num_frames, write_every=200):
        self.db = db
        self.config = config
        self.num_frames = num_frames
        self.num_complete = 0
        self.write_every = write_every

    def work(self, frame_id):
        frame_path = self.db.get_frame_path(frame_id)
        logger.debug(f'classifying dark with frame_id: {frame_id}')
        flags = pipe.classify_dark(frame_path, self.config['classify_dark'])
        is_good = flags.int_val == 0
        result = frame_id, flags.int_val, is_good
        return result

    def callback(self, result):
        self.num_complete += 1
        frame_id, flag_val, is_good = result
        self.db.update(frame_id, flags=flag_val, is_good=is_good)
        mod_num_complete = self.num_complete % self.write_every
        if mod_num_complete == 0 or self.num_complete == self.num_frames:
            frac_complete = self.num_complete / self.num_frames
            logger.debug(f'Completed {100 * frac_complete:.2f}% of frames')
            self.db.write_database(overwrite=True)


@utils.func_timer_info
def check_darks(args):
    config = utils.load_config(args.config_fn)
    db = DFIndividualFrames(config['db_path'], config['data_path'])
    dates = utils.parse_strings_or_file_to_list(args.dates, args.dates_fn) 

    date_q = [f"date=='{d}'" for d in dates]

    # pandas query breaks if the query is too long
    if len(date_q) <= 20:
        q = "frame_type=='dark' & (" + ' | '.join(date_q) + ')'
        frame_ids = db.query_index(q)
    else:
        frame_ids = []
        for d_q in  utils.divide_list_into_chunks(date_q, 20):
            q = "frame_type=='dark' & (" + ' | '.join(d_q) + ')' 
            frame_ids.append(db.query_index(q).values)
        frame_ids = np.hstack(frame_ids)

    logger.info(f"Classifying {len(frame_ids)} darks {' '.join(dates)}.")
    worker = Worker(db, config, len(frame_ids))
    utils.multiproc.work_it(worker, frame_ids, args.nproc, 'CHECK DARKS')


def main():
    args = utils.default_script_args()
    check_darks(args)
