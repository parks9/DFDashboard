"""
Functions for calculating the time and location of things.
"""
import datetime
import numpy as np
from scipy import stats
from astropy import units as u
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import get_sun, get_moon, SkyCoord, EarthLocation
from astropy.coordinates import Longitude, Latitude, AltAz

from .. import utils
from .. import logger
from .. import DFStruct


__all__ = ['calculate_air_mass', 
           'find_time_consensus',
           'get_header_radec',
           'get_today', 
           'get_moon_status', 
           'get_fov_corners',
           'get_image_corners', 
           'has_wcs', 
           'nms', 
           'to_skycoord', 
           'to_skycoord_list']


nms_lon = Longitude('-105:28:41', unit=u.deg)
nms_lat = Latitude('32:53:23', unit=u.deg)
nms_elevation = 7300 * u.imperial.ft
nms = EarthLocation.from_geodetic(nms_lon, nms_lat, nms_elevation)


def calculate_air_mass(skycoord, utc_time):
    """
    Assumes approximation from Hardie (1962).

    Parameters
    ----------
    skycoord : `~astropy.coordinates.SkyCoord`
        Coordinates of target.
    utc_time:
        UTC date / time of observations in YYYY-MM-DDThh:mm:ss format. This is
        the same formate that is in the fits headers. 

    Returns
    -------
    air_mass : float
        Air mass calculated using the approximation from Hardie (1962).
    """
    utc_time = Time(utc_time)
    nms_altaz = AltAz(obstime=utc_time, location=nms)
    target_nms = skycoord.transform_to(nms_altaz)
    secz = target_nms.secz.value
    air_mass = secz -  0.0018167 * (secz - 1) -\
                       0.0028750 * (secz - 1)**2 -\
                       0.0008083 * (secz - 1)**3
    return air_mass


def find_time_consensus(paths_or_headers, time_keyword='DATE', delta_max=5):
    """
    Find the most common time in headers to flag incorrect times. This is
    most useful when comparing images at fixed exposure time.

    Parameters
    ----------
    paths_or_headers : list of str or `~astropy.io.fits.Header`
        Paths or fits headers that should agree on the time.
    time_keyword : str (optional)
        Name of the time keyword in the headers.
    delta_max : `~astropy.units.Quantity` or float
        Maximum allowed time difference to be considered consistent with the
        consensus. If a float is given, the units will be assumed to be hours.

    Retuns
    ------
    deltas : `~astropy.time.TimeDelta`
        Differences with the most common time.
    mode : `~astropy.time.Time`
        Mode of the time distribution.
    on_time : ndarray with the same length as paths_or_headers
        Boolean mask where True values are consistent with the consensus.
    """
    headers = [utils.load_path_or_header(h) for h in paths_or_headers]
    times = Time([h[time_keyword] for h in headers])
    time_mode = stats.mode(times.to_value('unix')).mode[0]
    time_mode = Time(time_mode, format='unix')
    time_deltas = times - time_mode
    delta_max = utils.check_astropy_units(delta_max, 'hr')
    off_time = np.abs(time_deltas.to_value('hr') * u.hr) > delta_max
    time_mode = Time(time_mode.to_value('isot'))
    results = DFStruct(date=times, mode=time_mode, off_time=off_time)
    return results


def get_header_radec(path_or_header):
    """
    Get the ra and dec from a fits header.

    Parameters
    ----------
    path_or_header : str or `~astropy.io.fits.Header`
        Either the path to a fits file or an astropy header object.

    Returns
    -------
    ra : float
        Right Ascension.
    dec : float
        Declination.
    """
    header = utils.load_path_or_header(path_or_header)
    try:
        ra, dec = header['RA'], header['DEC']
    except KeyError:
        ra, dec = header['OBJCTRA'], header['OBJCTDEC']
    return ra, dec


def get_today():
    """Return today's date."""
    return datetime.date.today().strftime('%m/%d/%Y')
    

def get_moon_status(skycoord, utc_time):
    """
    Calculate properties of the moon.

    Parameters
    ----------`
    skycoord : `~astropy.coordinates.SkyCoord`
        Coordinates of target.
    utc_time:
        UTC date / time of observations in YYYY-MM-DDThh:mm:ss format. This is
        the same formate that is in the fits headers. 

    Returns
    -------
    alt : `~astropy.coordinates.Angle`
        Altitude of the moon.
    az : `~astropy.coordinates.Angle`
        Azimuth of the moon.
    phase : `~astropy.coordinates.Angle`
       Phase of the moon. 
    illumination : float
        Moon illumination.
    sep_from_target : `~astropy.coordinates.Angle`
        Angular separation between the moon and the target field.
    target_alt : `~astropy.coordinates.Angle`
        Altitude of target field.
    target_az : `~astropy.coordinates.Angle`
        Azimuth of target field.
    success : bool
        If True, moon status was successfully calculated. 
    """
    assert type(skycoord) == SkyCoord, 'Coords must be a SkyCoord object'
    try:
        utc_time = Time(utc_time)
        moon = get_moon(utc_time, nms)

        nms_altaz = AltAz(obstime=utc_time, location=nms)
        moon_nms = moon.transform_to(nms_altaz)
        target_nms = skycoord.transform_to(nms_altaz)
        sep_from_target = moon.separation(skycoord)

        sun = get_sun(utc_time)
        elongation = sun.separation(moon)
        phase = np.arctan2(sun.distance * np.sin(elongation),
                           moon.distance - sun.distance * np.cos(elongation))
        illumination = (1 + np.cos(phase.value)) / 2.0

        results = DFStruct(
            alt=moon_nms.alt,
            az=moon_nms.az,
            phase=phase,
            illumination=illumination,
            sep_from_target=sep_from_target,
            target_alt=target_nms.alt,
            target_az=target_nms.az,
            success=True
        )

    except Exception as e:
        logger.warning('Moon data calculation failed: {}'.format(e))
        results = DFStruct(success=False)
    return results


def get_fov_corners(radec_center, fov=[3, 2], scale=1):
    """
    Get the corners of FOV given the center coordinate.

    Parameters
    ----------
    radec_center : list of float or `~astropy.coordinates.SkyCoord`
        Central ra and dec.
    fov : list of float
        The field of view in degrees. 
    scale : float
        Scale FOV by this amount. 

    Returns
    -------
    fov : list of float
        The field of view in degrees.
    ra_min : float
        Minimum Right Ascension. 
    ra_max : float
        Maximum Right Ascension. 
    dec_min : float
        Minimum Declination. 
    dec_max : float
        Maximum Declination. 
    """
    if not utils.is_list_like(scale):
        scale = [scale, scale]
    sc = to_skycoord(radec_center)
    ra, dec = sc.ra.deg, sc.dec.deg
    fov_eff = scale[0] * fov[0] / np.cos(np.deg2rad(dec)), scale[1] * fov[1]
    ra_min, ra_max = ra - 0.5 * fov_eff[0], ra + 0.5 * fov_eff[0] 
    dec_min, dec_max = dec - 0.5 * fov_eff[1], dec + 0.5 * fov_eff[1] 
    bounds = dec_min, dec_max, ra_min, ra_max
    corners = DFStruct(fov=fov,
                       ra_min=ra_min, 
                       ra_max=ra_max, 
                       dec_min=dec_min, 
                       dec_max=dec_max, 
                       bounds=bounds, 
                       fov_eff=fov_eff)
    return corners


def get_image_corners(path_or_header):
    """
    Get the sky coordinates of the corners of an image taking SIP 
    corrections into account.

    Parameters
    ----------
    path_or_header : str or `~astropy.io.fits.Header`
        Path to fits file or astropy header object. 

    Returns 
    -------
    corners : (4, 2) array of (x, y) coordinates
        The order is clockwise starting with the bottom left corner.
    """
    header = utils.load_path_or_header(path_or_header)
    wcs = WCS(header)
    corners = wcs.calc_footprint()
    return corners


def has_wcs(path_or_header):
    """
    Check if header has a WCS.

    Parameters
    ----------
    path_or_header : str or `~astropy.io.fits.Header`
        Path to fits file or astropy header object. 

    Returns
    check : bool
        True if header has a WCS.
    """
    header = utils.load_path_or_header(path_or_header)
    check = 'WCSAXES' in header.keys()
    return check


def to_skycoord(coord):
    """
    Return SkyCoord of input coordinates (ra, dec).

    Parameters
    ----------
    coord : str or list or `~astropy.coordinates.SkyCoord`
        RA and DEC coordinates with flexible format options. 

    Returns
    -------
    sc : `~astropy.coordinates.SkyCoord`
        Astropy SkyCoord object with input RA and DEC.
    """
    if type(coord) == str:
        coord = coord.replace(',', ' ')
        coord = coord.split()
    if type(coord) == SkyCoord:
        sc = coord
    elif type(coord[0]) == str:
        assert type(coord[1]) == str, 'ra & dec must be in the same format!'
        coord = coord[0] + ' ' + coord[1]
        coord = coord.replace('h', ':').\
                      replace('d',':').\
                      replace('m',':').\
                      replace('s', '')
        sc =  SkyCoord(coord, frame='icrs', unit=(u.hourangle, u.deg))
    else:
        ra, dec = coord
        sc =  SkyCoord(ra, dec, unit='deg')
    return sc


def to_skycoord_list(list_of_coords):
    """
    Convert a list of RA and DEC coordinates in a flexible format to a 
    SkyCoord object.

    Parameters
    ----------
    list_of_coords : list of str or list or `~astropy.coordinates.SkyCoord`
        list of RA and DEC coordinates with flexible format options.

    Returns 
    -------
    sc : `~astropy.coordinates.SkyCoord`
        Astropy SkyCoord object with the input RA and DEC coordinates.
    """
    return SkyCoord(list(map(to_skycoord, list_of_coords)))
