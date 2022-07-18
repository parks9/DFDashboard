"""
Functions for doing neat things in Jupyter notebooks.

TODO: Make this work.
"""

import numpy as np
import pandas as pd


def print_classification_progress(frames_db, date=None, frame_type=None):
    """ Check progress towards classifying individual frames.

    Parameters
    ----------
    date : str or list (optional)
        Optionally restrict update to a single date or a range of dates.
        For a single date, enter the string: 'YYYY-MM-DD'
        For a range of dates, enter the start and end dates as a list of
        strings (i.e., ['YYYY-MM-DD', 'YYYY-MM-DD']).

    frame_type : str (optional)
        Optionally limit output to a particular type of frame
    """

    if frame_type is not None:
        title_str = '\nProgress: classifying '+frame_type.upper()+' frames'
    else:
        title_str = '\nProgress: classifying ALL framess'

    head_str = 'date\t\t num_frames \t num_checked \t Percent done'
    buffer_str1 = '\n'+'-'*(len(head_str)+18)+'\n'
    buffer_str2 = '\n'+'='*(len(head_str)+18)
    print(title_str, buffer_str1, head_str, buffer_str2)

    if date is None:
        date_list = frames_db.table.date.unique()
    else:
        if type(date) == list:
            date_range =  pd.date_range(start=date[0], end=date[1])
            date_list = [str(dt.date()) for dt in date_range]
        else:
            date_list = [date]

    reset_color_str = '\033[m' # reset to the defaults
    for date in date_list:
        mask_classified = frames_db.mask_database(date=date,
                                                  frame_type=frame_type,
                                                  is_good=True)
        mask_classified |= frames_db.mask_database(date=date,
                                                   frame_type=frame_type,
                                                   is_good=False)
        mask_all = frames_db.mask_database(date=date, frame_type=frame_type)

        num_checked = frames_db.table[mask_classified].is_good.count()
        num_total = frames_db.table[mask_all].is_good.count()

        if num_total == 0:
            percent_checked = 0.0 # avoid dividebyzero warnings
            font_color_str = '\33[31m' # Red Text

        else:
            percent_checked = np.around(100*num_checked/num_total, 2)

            if num_checked == num_total:
                font_color_str = '\033[32m' # Green Text
            else:
                font_color_str = reset_color_str

        print(font_color_str, date,'\t', num_total, '\t\t',
              num_checked, '\t\t', percent_checked,  reset_color_str)


def print_time_on_sky(frames_db, targets=None):
    """
    Check on-sky time for each target, split filter.

    Parameters
    ----------
    targets : str or list (optional)
        Optionally provide the name of a target (or a list of targets) to
        view. Otherwise all existing targets will be shown.

    NOTES
    -----
    The quoted times are NOT the total exposure time from all cameras
    combined, but rather the total time "on sky" that Dragonfly
    (considered as a single unit) spent on this object.

    We split by filters to indicate cases where the filter was unknown.
    """

    title_str = '\nMinutes on sky (48-lens equiv., no quality cuts)'
    head_str = 'Target\t\t'
    for fn in frames_db.table.filter_name.unique():
        head_str += fn+'\t'

    buffer_str1 = '\n'+'-'*(len(head_str)+28)+'\n'
    buffer_str2 = '\n'+'='*(len(head_str)+28)
    print(title_str, buffer_str1, head_str, buffer_str2)

    if targets is not None:
        if type(targets) is not list:
            targets = [targets]
        check_targets = targets

    else:
        check_targets = frames_db.targets

    grouped = frames_db.table.groupby(['target','filter_name'])
    for target in check_targets:
        if len(target) <= 8:
            out_str = target + '\t\t'
        else:
            out_str = target + '\t'

        for fn in frames_db.table.filter_name.unique():
            sec = grouped.get_group((target, fn)).sum().exptime
            minutes = sec / 60.
            min_48lenses = np.around(minutes / 48., 1)
            out_str += str(min_48lenses)+' \t'
        print(out_str)
