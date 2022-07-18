import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates.angle_utilities import angular_separation
from .. import DFStruct
from .. import utils


__all__ = ['pixel_area_map', 'pixel_area_map_brute_force']


def _pixel_sky_coordinates(path_or_header, edges=True):
    header = utils.load_path_or_header(path_or_header)
    if not utils.has_wcs(header):
        raise Exception(f'{path_or_header} does not have a WCS')
    shape = header['NAXIS2'], header['NAXIS1']
    yy, xx = np.indices(np.array(shape) + int(edges)) - 0.5 * edges
    coords = pixel_to_skycoord(xx, yy, WCS(header))
    return coords


def pixel_area_map(path_or_header, shape=None, blc=(1, 1), 
                   normalize_at='peak'):
    """
    Computes Pixel Area Map (PAM) using the distortion model defined in WCS
    and described through Simple Image Polynomials (SIP) by computing
    the Jacobian of the distortion model.

    This function computes the Jacobian of the distortion model using
    *symbolic differentiation* of Simple Image Polynomials.

    source:
    stsci-skypac.readthedocs.io/en/latest/_modules/stsci/skypac/pamutils.html

    Parameters
    ----------
    path_or_header : str or astropy.fits.Header
        Image path or fits header.
    shape : tuple of int, None, optional
        A tuple of two integers (ny, nx) indicating the size of the PAM image
        to be generated. When the default value is used (`None`), the size
        of the returned PAM array will be determined from ``wcs.array_shape``
        attribute of the supplied ``WCS`` object.
    blc : tuple of int or float, optional
        A tuple indicating the coordinates of the bottom-left pixel of the
        PAM array to be computed. These coordinates should be given
        in the image coordinate system defined by the input ``WCS`` (in which,
        for example, ``WCS.crpix`` is defined). The first element specifies
        the column (``"x"``-coordinate) and the second element specifies
        the row (``"y"``-coordinate).
    normalize_at : str or list or None (optional)
        Which pixel to normalize the image by. Options are 'center': use the 
        central pixel of the image, 'peak': use the pixel with the maximum 
        value, [row_i, col_j]: use the specified pixel, or None: do not 
        normalize the pam. 

    Returns
    -------
    pam : numpy.ndarray
        Pixel area map.
    """
    header = utils.load_path_or_header(path_or_header)
    if not utils.has_wcs(header):
        raise Exception(f'{path_or_header} does not have a WCS')
    wcs = WCS(header)


    if shape is None:
        shape = wcs.array_shape

    # distortion does not exist or is linear:
    if wcs.sip is None or wcs.sip.a_order < 1 or wcs.sip.b_order < 1 or \
       (wcs.sip.a_order == 1 and wcs.sip.b_order == 1):
        return np.ones(shape, dtype=np.float64)

    # prepare coordinates:
    x = np.arange(shape[1], dtype=np.float) - wcs.sip.crpix[0] + float(blc[0])
    y = np.arange(shape[0], dtype=np.float) - wcs.sip.crpix[1] + float(blc[1])

    ar = np.arange(wcs.sip.a_order + 1)
    br = np.arange(wcs.sip.b_order + 1)

    ones_a = np.ones(wcs.sip.a_order + 1)
    ones_b = np.ones(wcs.sip.b_order + 1)

    # "coordinate vectors" (e.g., (1, x, x**2, x**3, ...)) used in
    # distortion bilinear forms:
    ax = np.outer(x, ones_a)**ar
    ay = np.outer(y, ones_a)**ar
    bx = np.outer(x, ones_b)**br
    by = np.outer(y, ones_b)**br

    # derivatives of the "coordinate vectors" with regard to x & y:
    adx = np.roll(ax, 1, 1) * ar
    ady = np.roll(ay, 1, 1) * ar
    bdx = np.roll(bx, 1, 1) * br
    bdy = np.roll(by, 1, 1) * br

    # derivatives of the binomial forms:
    A = wcs.sip.a.T
    B = wcs.sip.b.T
    dadx = 1.0 + np.tensordot(ay.T, np.tensordot(A, adx, (1, 1)), (0, 0))
    dady = np.tensordot(ady.T, np.tensordot(A, ax, (1, 1)), (0, 0))
    dbdx = np.tensordot(by.T, np.tensordot(B, bdx, (1, 1)), (0, 0))
    dbdy = 1.0 + np.tensordot(bdy.T, np.tensordot(B, bx, (1, 1)), (0, 0))

    # compute rescaled Jacobian
    jacobian = np.abs(dadx * dbdy - dady * dbdx)

    if normalize_at is None:
        norm = 1
    elif utils.is_list_like(normalize_at):
        norm = jacobian[int(normalize_at[0]), int(normalize_at[1])]
    elif normalize_at == 'peak':
        norm = jacobian.max()
    elif normalize_at == 'center':
        slice_c = utils.slice_image_center(jacobian.shape)
        norm = jacobian[slice_c].mean()
    else:
        raise Exception(f'{normalize_at} is not a valid place to normalize.')
    pam = jacobian / norm

    return pam


def pixel_area_map_brute_force(path_or_header, normalize_at=None):
    """
    Generate the pixel area map by bute force.
    (i.e., calculate the area of each pixel)

    Parameters
    ----------
    path_or_header : str or astropy.fits.Header
        Image path or fits header.
    normalize_at : str or list or None (optional)
        Which pixel to normalize the image by. Options are 'center': use the 
        central pixel of the image, 'peak': use the pixel with the maximum 
        value, [row_i, col_j]: use the specified pixel, or None: do not 
        normalize the pam. 

    Returns
    -------
    pam : astropy.units.Quantity or ndarray
        The pixel area map. If normaal_at is None, will return an astropy 
        quantity with units of square arsec. Otherwise, will return a numpy 
        array that has been normalized.

    Notes
    -----
    This function was modified from the gammapy package:
    https://github.com/gammapy/gammapy/blob/master/gammapy/maps/wcs.py
    """
    coords = _pixel_sky_coordinates(path_or_header)

    # define pixel corners
    low_left = coords[..., :-1, :-1]
    low_right = coords[..., 1:, :-1]
    up_left = coords[..., :-1, 1:]
    up_right = coords[..., 1:, 1:]

    # compute side lengths
    low = low_left.separation(low_right)
    left = low_left.separation(up_left)
    up = up_left.separation(up_right)
    right = low_right.separation(up_right)

    # compute enclosed angles
    angle_low_right = low_right.position_angle(low_left) -\
                      low_right.position_angle(up_right) 
    angle_up_left = low_left.position_angle(up_left) -\
                    up_left.position_angle(up_right) 

    # compute area assuming a planar triangle
    area_low_right = 0.5 * low * right * np.sin(angle_low_right)
    area_up_left = 0.5 * up * left * np.sin(angle_up_left)

    pam = u.Quantity(area_low_right + area_up_left, 'sr').to('arcsec2')

    if normalize_at is not None:
        if normalize_at == 'peak':
            pam = pam.value / pam.value.max()
        elif normalize_at == 'center':
            slice_c = utils.slice_image_center(pam.shape)
            pam = pam.value / pam.value[slice_c].mean()
        elif utils.is_list_like(normalize_at):
            i, j = normalize_at
            pam = pam.value / pam.value[int(i), int(j)]
        else:
            raise Exception(f'{normalize_at} is not a valid normalization.')

    return pam
