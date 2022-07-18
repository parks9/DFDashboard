import os
from glob import glob
import numpy as np
import pandas as pd
from  tqdm import tqdm
from astropy.io import fits
from ..flags import FlatFlags, FlagArray
from ..tasks import flats as tasks
from .. import utils, logger, improc
from .. import DFStruct


default_accept_flags = ['MOON_UP', 'MOON_NEAR'] + FlatFlags.bookkeeping_strs
all_single_frame_flat_flags = ['negative_pixels', 'low_median_counts', 
                               'high_median_counts', 'bad_header_altaz', 
                               'moon_up', 'moon_near', 'bad_slope', 
                               'no_master_dark']


__all__ = ['assign_single_frame_flags', 'classify_flats']


def assign_single_frame_flags(frame_id_or_dict, db_hub, config={}):
    """ Assign all independent flags for a single image

    Parameters
    ----------
    frame_id_or_dict : int or dict or list
        Unique frame id number or dict with the required params to
        identify a frame uniquely.
    db_hub : dfreduce.DFDatabaseHub
        The database hub.
    config : dict (optional)
        Optionally pass in a dictionary containing information about the
        parameters for all relevant methods.

    Returns
    -------
    flags : str
        The flags. If no flags assigned, will be an empty string.
    """
    flags = FlatFlags()
    frame_id = db_hub.frames.check_frame_id(frame_id_or_dict)
    frame_type = db_hub.frames.df.loc[frame_id, 'frame_type']

    if frame_type != 'flat':
        msg = 'frame_type for frame {} = {}: Is this is what you want?'
        logger.warning(msg.format(frame_id, frame_type))

    _path = db_hub.frames.get_frame_path(frame_id)
    if utils.is_file_corrupt(_path).is_corrupt:
        flags.set('corrupt_file')
        return flags

    flat_ds = db_hub.get_dark_subtracted(frame_id)
    if flat_ds is None:
        flags.set('no_master_dark')
        return flags

    _config = config.copy()
    classify_flats = _config.pop('classify_flats', {})

    # check the number of negative pixels
    negative_pix_kw = classify_flats.pop('count_negative_pixels', {})
    flags += tasks.count_negative_pixels(flat_ds.pixels, **negative_pix_kw)

    # check fraction of pixels that are zero
    max_zero_frac = classify_flats.pop('max_zero_frac', 0.1)
    frac_zero = utils.zero_pixel_fraction(flat_ds.info.frame_path)
    if frac_zero > max_zero_frac:
        flags.set('ZERO_PIXEL_FRACTION')

    # make sure the median count level is reasonable
    med_counts_kw = classify_flats.pop('check_median_count_levels', {})
    flags += tasks.check_median_count_levels(flat_ds.pixels, **med_counts_kw)

    # was the moon up? how close to the FOV was it?
    moon_kw = classify_flats.pop('check_moon_proximity', {})
    flags += tasks.check_moon_proximity(flat_ds.header, **moon_kw)

    # stitch everything together
    f_msg = flags.to_string() if len(flags) > 0 else None
    logger.debug('flat {} independent flags = {}'.format(frame_id, f_msg))

    return flags


def create_twilight_master_flats(driver):
    """
    Create dusk and dawn master flats (where possible) for each camera.
    """
    logger.info('Creating dusk and dawn master flats for {}'.format(driver.date))

    accept_flags = default_accept_flags # TODO add this to config
    serialno = driver.db_hub.frames.query('serialno', index=driver.trunk.index)
    driver.trunk['serialno'] = serialno
    kws = driver.get_config('classify_flats').pop('create_master_flat', {})

    if 'single_frame_flags' not in driver.completed_steps:
        logger.critical('Single frame flags not assigned!')
        return None

    if 'mask_path' in kws.keys():
        kws['mask_path'] = os.path.join(kws['mask_path'], driver.date)
        utils.mkdir_if_needed(kws['mask_path'])

    driver.db_hub.mcals.setup_date_paths(driver.date)
    driver.trunk['has_master_flat'] = False
    grouped = driver.trunk.groupby(['serialno', 'part_of_day'])
    for (sn, pod) in grouped.groups:
        if pod == 'unknown':
            logger.critical(f'Part of day not known for {sn} flats.')
            continue

        log_str = 'Creating {} master flat for {} on {}'
        logger.debug(log_str.format(pod.upper(), sn, driver.date))

        idx = grouped.groups[(sn, pod)]
        f_arr = FlagArray('flat', flags=driver.trunk.loc[idx, 'flags'])
        keep = f_arr.count(ignore_flags=accept_flags) == 0
        keep_idx = idx[keep]

        num_eligible = len(keep_idx)
        logger.debug('Number of eligible flats: {}'.format(num_eligible))

        if len(keep_idx) > 0:
            mflat = tasks.create_master_flat(
                keep_idx, driver.db_hub, combinefunc=np.ma.mean, **kws)
        else:
            msg = 'No good flats for {} on {}'.format(sn, driver.date)
            logger.critical(msg)
            mflat = None

        if mflat is not None:
            out_fn = driver.db_hub.mcals.build_flat_path(driver.date, sn, pod)
            utils.write_pixels(out_fn, mflat.pixels, mflat.header)
            driver.trunk.loc[keep_idx, 'has_master_flat'] = True 


def group_assign_ramp_flags(driver):
    """ 
    Within a group of flats (same camera, either evening or morning),
    assess the quality (and uniformity) of the group members.
    """
    logger.info('Performing group ramp checks for {}'.format(driver.date))

    trunk = driver.trunk
    index = trunk.index[trunk.has_master_flat]
    classify_kw = driver.get_config('classify_flats')
    ff_kw = classify_kw.pop('flatten_flat', {})
    ramp_kw = classify_kw.pop('measure_ramp', {})
    max_ramp_stddev = classify_kw.pop('max_ramp_stddev', 5e-5)
    max_good_ramp_stddev = classify_kw.pop('max_good_ramp_stddev', 1e-6)

    msg = '{} flats on {} have a master flat'.format(len(index), driver.date)
    logger.info(msg)

    trunk['slope'] = pd.NA
    logger.info('Flat fielding {} flats on {}'.format(len(index), driver.date))
    for idx in index:
        flat_ds = driver.db_hub.get_dark_subtracted(idx)
        sn, pod = trunk.loc[idx,  ['serialno', 'part_of_day']].values
        path_id = dict(date=driver.date, serialno=sn, part_of_day=pod)
        master_flat_path = driver.db_hub.mcals.build_flat_path(**path_id)
        flat_ff = improc.divide_by_master_flat(flat_ds.pixels, 
                                               master_flat_path, 
                                               **ff_kw)
        ramp_results = tasks.measure_ramp(flat_ff.pixels, **ramp_kw)
        slope = ramp_results.slope
        trunk.loc[idx, 'slope'] = slope
        trunk.loc[idx, 'flags'] += ramp_results.flags.int_val
        flag_str = ramp_results.flags.to_string()
        msg = 'Slope = {:.2e}, flag = {} for frame {} on {}'
        logger.debug(msg.format(slope, idx, flag_str, driver.date))

    logger.info('Comparing ramp stddev of flat group on ' + driver.date)
    group_sn_pod = trunk.loc[index].groupby(['serialno', 'part_of_day'])

    for (sn, pod), data in group_sn_pod:
        f_obj = FlatFlags()
        idx = group_sn_pod.groups[(sn, pod)]
        log_str = 'Checking ramp variation within group: {} / {}'
        logger.debug(log_str.format(sn, pod.upper()))

        # if the stddev of ALL slopes in this group is too high, flag all
        if data.slope.std() > np.float(max_ramp_stddev):
            f_obj.set('bad_group,high_group_ramp_stddev')
        msg = 'Group ramp stddev: {}, flag: {}'
        logger.debug(msg.format(data.slope.std(), f_obj.to_string()))

        has_ramp_flag = FlagArray('flat', flags=data['flags']).is_set('BAD_SLOPE')
        if has_ramp_flag.sum() < len(data):
            # if the stddev of GOOD slopes in this group is too high, flag all
            good_group = data[~has_ramp_flag]
            if good_group.slope.std() > np.float(max_good_ramp_stddev):
                f_obj.set('bad_group,high_good_group_ramp_stddev')
            slope_stddev = good_group.slope.std()
        else:
            # if no good slopes exist at all in this group, flag all as bad
            f_obj.set('bad_group,no_good_ramps')
            slope_stddev = None

        trunk.loc[idx, 'flags'] += f_obj.int_val
        logger.debug(msg.format(slope_stddev, f_obj.to_string()))


def twilight_master_comparison(driver): 
    """ 
    Compare the dusk and dawn master flats for a given camera,
    if they both exist (and have not already been flagged as bad_group).
    If only one exists, choose a masterflat from another night to compare to.

    NOTES
    -----
    The flag 'moon_up' was considered safe for the creation of twilight group
    master flats, but is NOT allowed for the final master flats. This is why it is 
    important to check whether or not any ramps in the flattened master twilight 
    are due to the moon.
    """
    # group the flags table by serialno and flat_type
    trunk = driver.trunk
    index = trunk.index[trunk.has_master_flat]
    mcals = driver.db_hub.mcals

    classify_kw = driver.get_config('classify_flats')
    ff_kw = classify_kw.pop('flatten_flat', {})
    ramp_kw = classify_kw.pop('measure_ramp', {})
    group_sn = trunk.loc[index].groupby(['serialno'])

    for (sn, data) in group_sn:
        idx = group_sn.groups[sn]
        log_str = 'Comparing twilight masterflats within camera: {} '
        logger.debug(log_str.format(sn))

        morning = data[data.part_of_day == 'morning']
        f_arr = FlagArray('flat', flags=morning['flags'])
        morning_good = f_arr.is_set('BAD_GROUP').sum() == 0
        morning_good &= morning.has_master_flat.sum() > 0

        evening = data[data.part_of_day == 'evening']
        f_arr = FlagArray('flat', flags=evening['flags'])
        evening_good = f_arr.is_set('BAD_GROUP').sum() == 0
        evening_good &= evening.has_master_flat.sum() > 0

        f_obj = FlatFlags()
        if morning_good and evening_good:
            dawn_path = mcals.build_flat_path(driver.date, sn, 'morning')
            dusk_path = mcals.build_flat_path(driver.date, sn, 'evening')
            flat_ff = improc.divide_by_master_flat(dawn_path,
                                                   dusk_path, 
                                                   **ff_kw)
            ramp_flag = tasks.measure_ramp(flat_ff.pixels, **ramp_kw).flags
            if ramp_flag.count() > 0:
                f_obj.set('BAD_TWILIGHT_MASTER_SLOPE')
                f_arr = FlagArray('flat', flags=data['flags'])
                moon_mask = f_arr.is_set('moon_up')
                # blame the moon if we can
                if moon_mask.sum() > 0:
                    trunk.loc[idx[moon_mask], 'flags'] += f_obj.int_val
                else:
                    trunk.loc[idx, 'flags'] += f_obj.int_val
        else:
            # neither the evening nor the morning flats could be flattened.
            f_obj.set('SKIPPED_TWILIGHT_COMPARISON')
            trunk.loc[idx, 'flags'] += f_obj.int_val 
            msg = 'Could not flatten either twilight flat for {} on {}'
            logger.warn(msg.format(sn, driver.date))


def part_of_day_consensus(driver):
    """ 
    At fixed exposure number, all cameras should agree on whether or not 
    the moon is up, as well as on the time of day. This function makes 
    them all consistent. If less than frac_agree_thresh agree, then that 
    exposure number is flagged as confused -- each exposure is still made 
    consistent with the majority.
    """
    frac_agree_thresh = 0.5 # TODO: add to default config
    frame_db = driver.db_hub.frames
    part_of_day = frame_db.query_part_of_day(index=driver.trunk.index)
    flats = frame_db.query('expnum', index=driver.trunk.index)
    date = driver.date
    trunk = driver.trunk

    msg = 'Making part of day agree for {} flats on {} with thresh = {:.1f}%'
    logger.info(msg.format(len(flats), date, 100 * frac_agree_thresh))

    num_unknown = (part_of_day == 'unknown').sum()
    if num_unknown > 0:
        msg = '{} flats on {} have an unknown part of day'
        logger.warning(msg.format(num_unknown, date))

    # make sure things are consistent at fixed exposure number
    for expnum in flats.expnum.unique():
        expnum_mask = flats.expnum == expnum
        idx_e = expnum_mask.index[expnum_mask]
        consensus = part_of_day[expnum_mask].mode()[0]
        agree = part_of_day[expnum_mask] == consensus
        disagree = ~agree
        frac_agree = agree.sum() / len(agree)
        if frac_agree < frac_agree_thresh:
            flag = 'PART_OF_DAY_CONFUSION'
            msg = '<{:.1f}% of flats on {} exp = {} agree on part of day!'
            logger.critical(msg.format(100 * frac_agree, date, expnum))
            trunk.loc[idx_e, 'flags'] += FlatFlags.str_to_int[flag]
        msg = '{:.1f}% of flats on {} agree exp = {} occured in the {}'
        logger.debug(msg.format(100 * frac_agree, date, expnum, consensus))

        # the part of day should agree between all cameras
        if disagree.sum() > 0:
            flag = 'CHANGED_PART_OF_DAY'
            idx = idx_e[disagree]
            trunk.loc[idx, 'flags'] += FlatFlags.str_to_int[flag]
            part_of_day.loc[idx] = consensus
    trunk['part_of_day'] = part_of_day


def moon_consensus(driver):
    frac_agree_thresh = 0.5 # TODO: add to default config
    frame_db = driver.db_hub.frames
    flats = frame_db.query('expnum').loc[driver.trunk.index]
    date = driver.date
    trunk = driver.trunk
    if 'single_frame_flags' not in driver.completed_steps:
        logger.error('Single frame flags must be assigned before moon checks')
        return None

    msg = 'Making moon status agree for {} flats on {} with thresh = {:.1f}%'
    logger.info(msg.format(len(flats), date, 100 * frac_agree_thresh))

    # make sure things are consistent at fixed exposure number
    for expnum in flats.expnum.unique():
        # all cameras should agree about the moon
        expnum_mask = flats.expnum == expnum
        idx_e = expnum_mask.index[expnum_mask]
        f_arr = FlagArray('flat', flags=trunk.loc[idx_e]['flags'])
        moon_up = f_arr.is_set('MOON_UP')
        consensus =  moon_up.sum() > 0.5 * len(moon_up)
        agree = moon_up == consensus
        frac_agree = agree.sum() / len(moon_up)
        disagree = ~agree
        if frac_agree < frac_agree_thresh:
            flag = 'MOON_CONFUSION'
            msg = '<{:.1f}% of flats on {} exp = {} agree about the moon!'
            logger.critical(msg.format(100 * frac_agree, date, expnum))
            trunk.loc[idx_e, 'flags'] += FlatFlags.str_to_int[flag]
        up_or_down = 'up' if consensus else 'down'
        msg = '{:.1f}% of flats on {} exp = {} agree the moon was {}' 
        logger.debug(msg.format(100 * frac_agree, date, expnum, up_or_down))

        idx = idx_e[disagree]
        changed_flag = 'CHANGED_MOON_STATUS'
        if disagree.sum() > 0 and up_or_down == 'up':
            trunk.loc[idx, 'flags'] += FlatFlags.str_to_int['MOON_UP']
            trunk.loc[idx, 'flags'] += FlatFlags.str_to_int[changed_flag]
            msg = 'changed moon from down/unknown to up for {} flats on {}'
            logger.debug(msg.format(disagree.sum(), date))
        elif disagree.sum() > 0 and up_or_down == 'down':
            trunk.loc[idx, 'flags'] -= FlatFlags.str_to_int['MOON_UP']
            trunk.loc[idx, 'flags'] += FlatFlags.str_to_int[changed_flag] 
            msg = 'changed moon from up/unknown to down for {} flats on {}'
            logger.debug(msg.format(disagree.sum(), date))

    
def cross_camera_comparison(driver):
    """ 
    Check to see if we have missed any subtle systematic problems.
    Compare at fixed *exposure number* determine the fraction of images 
    that have been flagged. If this fraction is too high, flag everything.
    """
    logger.info('Performing cross-camera flag comparison for ' + driver.date)

    max_flagged_frac = 0.5 # TODO: add this to config

    frame_db = driver.db_hub.frames
    flats = frame_db.query('expnum').loc[driver.trunk.index]
    group_expnum = flats.groupby('expnum')

    for expnum, data in group_expnum:
        f_arr = FlagArray('flat', flags=driver.trunk.loc[data.index, 'flags'])
        count = f_arr.count(ignore_flags=default_accept_flags)
        has_flags = (count > 0).astype(float)
        frac_with_flags = has_flags.sum() / len(data)
        if frac_with_flags > max_flagged_frac:
            f_val = FlatFlags.str_to_int['BAD_BY_ASSOC']
            driver.trunk.loc[data.index, 'flags'] += f_val


def final_classification(driver):
    """ 
    For every image, combine all flags that have been assigned to it,
    and determine whether it qualifies as "good" or not.
    """
    logger.info('Performing final flat classification for ' + driver.date)
    f_arr = FlagArray('flat', flags=driver.trunk['flags'])
    count = f_arr.count(ignore_flags=default_accept_flags)
    driver.trunk['is_good'] = False
    driver.trunk.loc[count == 0, 'is_good'] = True


@utils.func_timer_info
def classify_flats(date, driver, run_label=None, **kwargs):
    """ 
    Pipeline to classify flat frames on a given night.

    Parameters
    ----------
    date : str
        Date to classify. YYYY-MM-DD.
    driver : dfreduce.Driver
        The pipeline driver.
    run_label : str (optional)
        Unique label for the checkpoint file.

    Notes
    -----
    The driver's trunk will be updated with the flags and classifications.
    """
    checkpoint_path = driver.checkpoints_path
    utils.mkdir_if_needed(checkpoint_path)

    label = '' if run_label is None else '_' + run_label
    fn = 'classify-flats-{}{}.pkl'.format(date, label)
    checkpoint_fn = os.path.join(checkpoint_path, fn)

    if driver.stuff_in_trunk:
        logger.info('Stuff found in trunk... starting from where you left off')
        logger.info('Completed steps: ' + ', '.join(driver.completed_steps))
        index = driver.trunk.index
    else:
        kwargs.update(dict(date=date, frame_type='flat'))
        index = driver.db_hub.frames.query_index(date=date, frame_type='flat')
        driver.create_empty_trunk(index)
        driver.add_attribute('date', date)
        driver.trunk['flags'] = pd.NA
        if driver.db_hub.frames.loc[index, 'flags'].notna().sum() > 0:
            msg = 'Flat flags for {} found in database -- Will overwrite'
            logger.warning(msg.format(date))
            driver.db_hub.frames.loc[index, 'flags'] = pd.NA
            driver.db_hub.frames.write_database(overwrite=True)

    if not driver.is_complete('single_frame_flags'):
        msg = 'for {} flats on {}'.format(len(index), date)
        logger.info('Assigning independent flags ' + msg)
        logger.start_tqdm()
        for idx in tqdm(index):
            flags = assign_single_frame_flags(
                idx, driver.db_hub, driver.get_config())
            driver.trunk.loc[idx, 'flags'] = flags.int_val
        logger.end_tqdm()
        driver.add_completed_step('single_frame_flags', checkpoint_fn)

    if not driver.is_complete('part_of_day_consensus'):
        part_of_day_consensus(driver)
        driver.add_completed_step('part_of_day_consensus', checkpoint_fn)

    if not driver.is_complete('moon_consensus'):
        moon_consensus(driver)
        driver.add_completed_step('moon_consensus')

    if not driver.is_complete('create_twilight_master_flats'):
        create_twilight_master_flats(driver)
        driver.add_completed_step('create_twilight_master_flats', checkpoint_fn)

    if not driver.is_complete('group_assign_ramp_flags'):
        group_assign_ramp_flags(driver)
        driver.add_completed_step('group_assign_ramp_flags', checkpoint_fn)

    if not driver.is_complete('twilight_master_comparison'):
        twilight_master_comparison(driver)
        driver.add_completed_step('twilight_master_comparison', checkpoint_fn)

    if not driver.is_complete('cross_camera_comparison'):
        cross_camera_comparison(driver)
        driver.add_completed_step('cross_camera_comparison', checkpoint_fn)

    if not driver.is_complete('final_classification'):
        final_classification(driver)
        driver.add_completed_step('final_classification', checkpoint_fn)

    if driver.delete_twilight_flats:
        logger.info('Deleting twilight diagnostic flats')
        path = os.path.join(driver.db_hub.mcals.flat_path, date)
        files = glob(os.path.join(path, '*evening.fits')) +\
                glob(os.path.join(path, '*morning.fits')) +\
                glob(os.path.join(path, '*unknown.fits'))
        for fn in files:
            logger.debug('Deleting ' + fn)
            os.remove(fn)
