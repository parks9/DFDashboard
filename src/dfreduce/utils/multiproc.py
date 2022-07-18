# Standard library
from functools import partial
from multiprocessing import Pool

# Third-party
from tqdm import tqdm

# Project
from .. import utils, logger


__all__ = ['work_it', 'thread_it']


def work_it(worker, loop_variable, nproc=1, name='job'):
    
    if nproc == 1:
        logger.start_tqdm()
        for var in tqdm(loop_variable):
            var = utils.parse_to_list(var)
            results = worker.work(*var)
            worker.callback(results)
        logger.end_tqdm()
    else:
        logger.info(f'Starting {nproc} processes to run {name}.')

        pool = Pool(processes=nproc)

        jobs = [pool.apply_async(worker.work, (var, ),\
                callback=worker.callback) for var in loop_variable]

        pool.close()

        logger.start_tqdm()
        for job in tqdm(jobs):
            job.get()
        logger.end_tqdm()

        logger.info(f'Finished running {name}. Now woking on single core.')


def thread_it(loop_func, loop_variable, nproc=1, name='job', initializer=None, 
              initargs=None, **kwargs):

    if nproc == 1:
        logger.start_tqdm()
        for var in tqdm(loop_variable):
            vars = utils.parse_to_list(var)
            loop_func(*vars, **kwargs)
        logger.end_tqdm()
    else:
        logger.info(f'Starting {nproc} processes to run {name}.')

        pool = Pool(processes=nproc, 
                    initializer=initializer, 
                    initargs=initargs)

        func = partial(loop_func, **kwargs)
        if utils.is_list_like(loop_variable[0]):
            jobs = [pool.apply_async(func, args) for args in loop_variable]
        else:
            jobs = [pool.apply_async(func, (var, )) for var in loop_variable]
        pool.close()

        logger.start_tqdm()
        for job in tqdm(jobs):
            job.get()
        logger.end_tqdm()

        logger.info(f'Finished running {name}. Now woking on single core.')
