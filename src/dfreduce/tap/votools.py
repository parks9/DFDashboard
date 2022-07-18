import numpy as np
from astropy import units as u
from .. import utils, logger


__all__ = ['circle_centers']


def circle_centers(path_or_header=None, image_corners=None, radius=0.25*u.deg):
    """
    Return the coordinates of circles of specified 
    radius needed to tile a region on the sky.
    """
    if path_or_header is not None:
        image_centers = utils.get_image_corners(path_or_header)
    else:
        msg = 'You must provide path_or_header *or* image_corners'
        assert image_corners is not None, msg
    
    ra_min, dec_min = image_corners.min(axis=0)
    ra_max, dec_max = image_corners.max(axis=0)
    
    circle_sep = np.sqrt(2) * radius.to('deg').value
    ra_arr = np.arange(ra_min, ra_max + circle_sep/2, circle_sep)
    dec_arr = np.arange(dec_min, dec_max + circle_sep/2, circle_sep)
    
    centers = []
    for ra in ra_arr:
        for dec in dec_arr:
            centers.append([ra, dec])
    centers = np.array(centers)

    return centers
