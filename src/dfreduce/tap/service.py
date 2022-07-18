import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import unique as unique_rows
from astropy.table import vstack
from tqdm import tqdm
from .. import utils, logger
from . import votools


try:
    import pyvo
except:
    logger.warning('pyvo not found: no TAP service for you')


__all__ = ['GaiaTAP', 'PanstarrsTAP']


class TAPBase(object):
    """
    Base class for accessing databases using Virtual Observatory 
    and the Table Access Protocol (TAP). Specific databases should 
    be created as subclasses that inherit this base class.
    """
    name = None
    tap_service_url = None
    default_query_template = None
    default_select = None

    def __init__(self):
        self.name = self.name
        if self.tap_service_url is not None:
            self.tap = pyvo.dal.TAPService(self.tap_service_url)

    def _check_query(self, s, q):
        if s is None:
            s = self.default_select 
        if q is None:
            q = self.default_query_template
        return s, q

    def send_query(self, query):
        result = self.tap.run_sync(query)
        return result.to_table()

    def quey_region(self, ra_c, dec_c, radius, select=None, 
                    query_template=None):
        radius = utils.check_astropy_units(radius, u.deg)
        select, query_template = self._check_query(select, query_template)
        q = query_template.format(select, ra_c, dec_c, radius.to('deg').value)
        result_table = self.send_query(q)
        return result_table

    def query_footprint(self, path_or_header, select=None, query_template=None, 
                        patch_radius=0.25*u.deg):
        patch_radius = utils.check_astropy_units(patch_radius, u.deg)
        self.header = utils.load_path_or_header(path_or_header)
        self.image_corners = utils.get_image_corners(self.header)
        select, qtemp = self._check_query(select, query_template)
        centers = votools.circle_centers(image_corners=self.image_corners, 
                                         radius=patch_radius)
        cat = []
        logger.start_tqdm()
        msg = f'Sending {len(centers)} queries to {self.name} TAP service'
        logger.info(msg)
        for c in tqdm(centers):
            r = patch_radius.to('deg').value
            tab = self.quey_region(c[0], c[1], r, select, qtemp)
            cat.append(tab)
        logger.end_tqdm()
        cat = unique_rows(vstack(cat))
        return cat


class GaiaTAP(TAPBase):
    name = 'Gaia'
    tap_service_url = 'https://gea.esac.esa.int/tap-server/tap'
    default_query_template = """SELECT {} FROM gaiadr2.gaia_source
    WHERE CONTAINS(POINT('ICRS', ra, dec),CIRCLE('ICRS', {}, {}, {}))=1
    """
    default_select = '*'


class PanstarrsTAP(TAPBase):
    name = 'PanSTARRS'
    tap_service_url = 'http://vao.stsci.edu/PS1DR2/tapservice.aspx'
    default_query_template = """SELECT {} FROM dbo.MeanObjectView
    WHERE CONTAINS(POINT('ICRS', raMean, decMean),CIRCLE('ICRS', {}, {}, {}))=1
    AND ng > 5
    AND nr > 5
    AND ni > 5
    AND nz > 5
    AND ny > 5
    AND rMeanPSFMag > -999
    AND iMeanPSFMag > -999
    AND zMeanPSFMag > -999
    AND yMeanPSFMag > -999
    AND gMeanPSFMag > 10
    AND gMeanPSFMag < 23"""
    default_select = 'raMean,decMean,raMeanErr,decMeanErr,nDetections,' 
    default_select += ','.join([f'n{b},{b}MeanPSFMag' for b in 'grizy']) + ','
    default_select += ','.join([f'{b}MeanPSFMagErr,{b}Flags' for b in 'grizy'])
