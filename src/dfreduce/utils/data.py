"""
Functions for fetching non-image data (e.g., the UW fields).
"""
import os
import pandas as pd
from astropy.table import Table
from .coordinates import to_skycoord
from .. import DFStruct
from .. import project_dir


__all__ = ['atm_ext_coeff', 
           'survey_radec_fetcher',
           'fetch_fiducial_zp', 
           'fetch_ultrawide_fields', 
           'fetch_ultrawide_field_radec', 
           'fetch_m33_field_radec']


atm_ext_coeff = DFStruct(g=-0.17, r=-0.11, V=-0.14)


def fetch_fiducial_zp(serialno):
    fn = os.path.join(project_dir, 'data/fiducial_zero_points.csv')
    zps = pd.read_csv(fn, index_col='serialno')
    return zps.loc[serialno, 'zp']


def fetch_ultrawide_fields():
    fn = os.path.join(project_dir, 'data/dfuw-fields.csv')
    return pd.read_csv(fn, index_col='field')


def fetch_ultrawide_field_radec(field, as_skycoord=True):
    uw = fetch_ultrawide_fields()
    if type(field) == str:
        field = int(field[2:])
    radec = uw.loc[field, ['ra', 'dec']].tolist()
    if as_skycoord:
        radec = to_skycoord(radec)
    return radec


def fetch_m33_field_radec(field, as_skycoord=True):
    fields = Table.read(
    """field ra dec
       M33_1 01:28:23.52 +28:17:58.79
       M33_2 01:28:23.52 +26:35:22.79
       M33_3 01:38:00.00 +26:35:22.79
       M33_4 01:38:00.00 +24:52:46.79
    """,
    format='ascii'
    )
    field = field if type(field) == str else f'M33_{field}'
    ra, dec = fields[fields['field'] == field]['ra', 'dec'][0]
    radec = ra + ' ' + dec
    if as_skycoord:
        radec = to_skycoord(radec)
    return radec


def fetch_LE_fields():
    fn = os.path.join(project_dir, 'data/lightecho-fields.csv')
    return pd.read_csv(fn, index_col='field')


def fetch_LE_field_radec(field, as_skycoord=True):
    le = fetch_LE_fields()

    # find ra, dec of field
    radec = le.loc[le['name']==field, ['ra', 'dec']].iloc[0].tolist()
    if as_skycoord:
        radec = to_skycoord(radec)
    return radec


def fetch_DD_targets():
    fn = os.path.join(project_dir, 'data/dd-targets.csv')
    return pd.read_csv(fn, index_col='field')


def fetch_DD_target_radec(field, as_skycoord=True):
    dd = fetch_DD_targets()

    # find ra, dec of field
    radec = dd.loc[dd['name']==field, ['ra', 'dec']].iloc[0].tolist()
    if as_skycoord:
        radec = to_skycoord(radec)
    return radec


survey_radec_fetcher = dict(
    UW=fetch_ultrawide_field_radec,
    M33=fetch_m33_field_radec,
    LE=fetch_LE_field_radec,
    DD=fetch_DD_target_radec
)
