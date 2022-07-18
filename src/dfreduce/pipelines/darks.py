from copy import deepcopy
from .. import utils, logger
from ..flags import DarkFlags
from ..tasks import check_stats, check_column_fits, check_zero_pixel_fraction


@utils.func_timer_debug
def classify_dark(path_or_pixels, config={}):
    """
    Classify a single dark frame.

    Parameters
    ----------
    path_or_pixels : str or ndarray
        Full path to dark frame or thie image pixels.
    config : dict
        Options for each tasks. Must have the following form:
        config = dict(check_stats={...}, check_column_fits={...})

    Returns
    -------
    flags : dfreduce.flags.DarkFlags
        Flags from the check_stats and check_column_fits tasks.
    """
    flags = DarkFlags()

    _config = deepcopy(config)

    results = utils.is_file_corrupt(path_or_pixels)

    if results.is_corrupt:
        flags.set('CORRUPT_FILE')
    else:
        pixels = results.pixels
        stats_kw = _config.pop('check_stats', {})
        flags += check_stats(pixels.copy(), **stats_kw).flags

        fits_kw = _config.pop('check_column_fits', {})
        flags += check_column_fits(pixels.copy(), **fits_kw).flags

        zero_pix_kw = _config.pop('check_zero_pixel_fraction', {})
        flags += check_zero_pixel_fraction(pixels.copy(), **zero_pix_kw).flags

    return flags
