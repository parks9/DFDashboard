import os, shutil
from glob import glob
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.utils.decorators import lazyproperty
from .cameras import camera_info, get_filter_name
from .log import EmptyImageError
from . import logger
from . import utils


__all__ = ['ImageButler', 'ImageCollection']
frame_types = ['flat', 'dark', 'light']
exo_targets = [
    "TOI",
    "CTOI",
    "EXO","exo","Exo","ExoDark",
    "WASP","Wasp",
    "HAT",
    "K2",
    "CoRoT","COROT","corot","Corot",
    "KELT",
    "UNSPECIFIED",
    "Merrin_star"
]


def _setup_exo_path(fn, temp_exo_path='/media/dragonfly/NAS/TempEXO'):
    rel_path = '/'.join(os.path.dirname(fn).split('/')[-2:])
    date = rel_path.split('/')[1]
    df_path = rel_path.split('/')[0]
    path = os.path.join(temp_exo_path, df_path)
    utils.mkdir_if_needed(path)
    path = os.path.join(temp_exo_path, df_path, date)
    utils.mkdir_if_needed(path)
    return path


def _check_df_unit_name(name):
    if type(name) == int:
        name = 'Dragonfly' + str(name)
    return name


class ImageCollection(object):

    def __init__(self, df_unit_files, survey='UW', clean_exo=True):
        self.survey = survey

        self.path = os.path.dirname(df_unit_files[0])
        self.date = self.path.split('/')[-1]
        self.df_unit = self.path.split('/')[-2]
        self.df_unit_path = os.path.dirname(self.path)
        self.clean_exo = clean_exo
        self.num_exo_files_removed = 0

        if self.survey == 'NB':
            self.add_file_info_flex(df_unit_files)
        else:
            self.add_file_info(df_unit_files)

    def add_file_info(self, df_unit_files):
        self.files = []
        self.corrupt_files = []
        self.exptimes = []
        self.filters = []
        self.targets = []
        self.datetime = []
        self.ra = []
        self.dec = []
        self.alt = []
        self.az = []
        self.ccd_temp = []
        self.expnum = []
        for fn in df_unit_files:
            try:
                head = fits.getheader(fn)
                if head['NAXIS1'] == 0 or head['NAXIS2'] == 0:
                    raise EmptyImageError(f'{fn} is empty.')
                if self.clean_exo:
                    fn, is_exo = self.check_if_exo(fn, head)
                    if is_exo:
                        self.num_exo_files_removed += 1
                        continue
                self.exptimes.append(head['exptime'])
                self.filters.append(get_filter_name(head))
                self.targets.append(head['TARGET'])
                self.datetime.append(head['DATE'])
                self.ra.append(head['OBJCTRA'])
                self.dec.append(head['OBJCTDEC'])
                self.alt.append(head['ALTITUDE'])
                self.az.append(head['AZIMUTH'])
                self.ccd_temp.append(head['TEMPERAT'])
                self.expnum.append(int(fn.split('/')[-1].split('_')[1]))
                self.files.append(fn)
            except (OSError, KeyError, EmptyImageError) as e:
                logger.warning('{}|{}: {}'.format(self.df_unit, self.date, e))
                logger.warning('{}|{}: Will not make exposure time array for'
                               ' {}'.format(self.df_unit, self.date, fn))
                self.corrupt_files.append(fn)

        self.files = np.array(self.files)
        self.exptimes = np.array(self.exptimes)
        self.filters = np.array(self.filters)
        self.targets = np.array(self.targets)
        self.datetime = np.array(self.datetime)
        self.ra = np.array(self.ra)
        self.dec = np.array(self.dec)
        self.alt = np.array(self.alt)
        self.az = np.array(self.az)
        self.ccd_temp = np.array(self.ccd_temp)
        self.expnum = np.array(self.expnum)

    def add_file_info_flex(self, df_unit_files):
        file_info = pd.DataFrame(index=df_unit_files,
                                 columns=['good','exptimes','filters',
                                          'targets','datetime','ra','dec',
                                          'alt','az','ccd_temp','expnum',
                                          'tilt','rawtilt','corrtilt',
                                          'tiltgoal'])

        logger.debug(file_info.index)

        for fn in df_unit_files:
            try:
                head = fits.getheader(fn)
                file_info.loc[fn]['exptimes']=head['exptime']
                filtkey = 'FILTER' if 'FILTER' in head.keys() else 'FILTNAM'
                thisfilter = '' if filtkey not in head.keys() else head[filtkey]
                file_info.loc[fn]['filters']=thisfilter
                target = head['TARGET'] if 'TARGET' in head.keys() else 'UNKNOWN'
                file_info.loc[fn]['targets']=target
                datekey = 'DATE' if 'DATE' in head.keys() else 'DATE-OBS'
                file_info.loc[fn]['datetime']=head[datekey]
                rakey = 'OBJCTRA' if 'OBJCTRA' in head.keys() else 'RA'
                file_info.loc[fn]['ra']=head[rakey]
                deckey = 'OBJCTDEC' if 'OBJCTDEC' in head.keys() else 'DEC'
                file_info.loc[fn]['dec']=head[deckey]
                altkey = 'ALTITUDE' if 'ALTITUDE' in head.keys() else 'CENTALT'
                file_info.loc[fn]['alt']=head[altkey]
                azimuth = head['AZIMUTH'] if 'AZIMUTH' in head.keys() else ''
                file_info.loc[fn]['az']=azimuth
                tempkey = 'TEMPERAT' if 'TEMPERAT' in head.keys() else 'CCD-TEMP'
                thistemp = 99.99 if tempkey not in head.keys() else head[tempkey]
                file_info.loc[fn]['ccd_temp']=thistemp
                try:
                    thisexpnum = int(fn.split('/')[-1].split('_')[1])
                except ValueError:
                    logger.warning(f'Unable to determine expnum for {fn}.')
                    thisexpnum = int(0)
                file_info.loc[fn]['expnum']=thisexpnum
                if self.survey=='NB':
                    if 'TILT' in head.keys():
                        file_info.loc[fn]['tilt']=head['TILT']
                    else:
                        file_info.loc[fn]['rawtilt']=head['RAWTILT']
                        file_info.loc[fn]['corrtilt']=head['CORRTILT']
                        file_info.loc[fn]['tiltgoal']=head['TILTGOAL']
                file_info.loc[fn]['good']=True
            except (OSError, KeyError) as e:
                logger.warning('{}|{}: {}'.format(self.df_unit, self.date, e))
                logger.warning('{}|{}: Will not make exposure time array for'
                               ' {}'.format(self.df_unit, self.date, fn))
                file_info.loc[fn]['good']=False

        self.corrupt_files = np.array(file_info[file_info['good']!=True])

        file_info = file_info[file_info['good']==True]

        self.files = np.array(file_info.index)
        self.exptimes = np.array(file_info.exptimes)
        self.filters = np.array(file_info.filters)
        self.targets = np.array(file_info.targets)
        self.datetime = np.array(file_info.datetime)
        self.ra = np.array(file_info.ra)
        self.dec = np.array(file_info.dec)
        self.alt = np.array(file_info.alt)
        self.az = np.array(file_info.az)
        self.ccd_temp = np.array(file_info.ccd_temp)
        self.expnum = np.array(file_info.expnum)
        self.tilt = np.array(file_info.tilt)
        self.rawtilt = np.array(file_info.rawtilt)
        self.corrtilt = np.array(file_info.corrtilt)
        self.tiltgoal = np.array(file_info.tiltgoal)

    def file_names(self, exptime=None, tag=None):
        if exptime is not None:
            exptime_mask = self.exptimes == exptime
        else:
            exptime_mask = np.ones(len(self.files), dtype=bool)
        if tag is not None:
            tag_mask = [tag in os.path.basename(f) for f in self.files]
            tag_mask = np.array(tag_mask, dtype=bool)
        else:
            tag_mask = np.ones(len(self.files), dtype=bool)
        files = self.files[tag_mask & exptime_mask]
        return files

    def fetch_all_pixels(self, exptime=None, tag=None):
        files = self.file_names(exptime, tag)
        pixels = [fits.getdata(f) for f in files]
        return pixels

    def fetch_all_headers(self, exptime=None, tag=None):
        files = self.file_names(exptime, tag)
        headers = [fits.getheader(f) for f in files]
        return headers

    def fetch_single_header(self, exptime=None, tag=None, idx=0):
        files = self.file_names(exptime, tag)
        return fits.getheader(files[idx])

    def count(self, exptime=None, tag=None):
        return len(self.file_names(exptime, tag))

    @lazyproperty
    def relative_dates(self):
        dates = [d for d in os.listdir(self.df_unit_path) if d[:2]=='20']
        dates = np.array(dates, dtype=np.datetime64)
        diffs = np.abs(dates - np.datetime64(self.date))
        argsort = np.argsort(diffs)
        sorted_relative_dates = dates[argsort]
        return sorted_relative_dates

    def check_if_exo(self, fn, header, **kwargs):
        is_exo = False
        base_fn = os.path.basename(fn)
        if '_EXO' in base_fn:
            exo_path = _setup_exo_path(fn, **kwargs)
            new_fn = os.path.join(exo_path, base_fn)
            if header['TARGET'] == 'Dark':
                msg = f'COPYING {fn} --> {new_fn}'
                logger.warning(f'COPYING {fn} --> {new_fn}')
                shutil.copy(fn, new_fn)
                logger.warning(f'REMOVING "EXO" tag in {fn}') 
                os.rename(fn, fn.replace('_EXO.fits', '.fits'))
                fn = fn.replace('_EXO.fits', '.fits')
            else:
                logger.warning(f'MOVING {fn} --> {new_fn}')
                shutil.move(fn, new_fn)
                is_exo = True
        elif any(t in header['TARGET'] for t in exo_targets):
            exo_path = _setup_exo_path(fn, **kwargs)
            new_fn = os.path.join(exo_path, base_fn)
            logger.warning(f'EXO target found: MOVING {fn} --> {new_fn}')
            shutil.move(fn, new_fn)
            is_exo = True
        return fn, is_exo


class ImageButler(object):

    def __init__(self, base_path, survey='UW'):

        self.survey = survey
        self.base_path = base_path

    def df_unit_path(self, df_unit, date=None):
        df_unit = _check_df_unit_name(df_unit)
        if date is None:
            path = os.path.join(self.base_path, df_unit)
        else:
            path = os.path.join(self.base_path, os.path.join(df_unit, date))
        return path

    def df_unit_files(self, df_unit, date, tag=None, tag_path=None):
        df_name = _check_df_unit_name(df_unit)
        date_path = self.df_unit_path(df_unit, date)
        serialno = camera_info[self.survey]['serialno_dict'][df_name]
        glob_str = f'{serialno}*.fits'
        if tag_path is not None:
            files = glob(os.path.join(date_path, tag_path, glob_str))
        else:
            files = glob(os.path.join(date_path, glob_str))
        if tag is not None:
            files = [f for f in files if tag in os.path.basename(f)]
        return files

    def glob_files(self, date, tag=None, tag_path=None):
        if tag_path is not None:
            files = glob(os.path.join(self.base_path, date, tag_path, '*.fits'))
        else:
            files = glob(os.path.join(self.base_path, date, '*.fits'))
        if tag is not None:
            files = [f for f in files if tag in os.path.basename(f)]
        return files

    def relative_dates(self, df_unit, date):
        dirs = os.listdir(self.df_unit_path(df_unit))
        dates = [d for d in dirs if d[:2]=='20']
        dates = np.array(dates, dtype=np.datetime64)
        diffs = np.abs(dates - np.datetime64(date))
        argsort = np.argsort(diffs)
        sorted_relative_dates = dates[argsort]
        return sorted_relative_dates

    def fetch_frames(self, df_unit, date, frame_type, tag=None, target=None):
        df_unit = _check_df_unit_name(df_unit)
        files = self.df_unit_files(df_unit, date)
        if frame_type not in frame_types:
            raise Exception(frame_type + ' is not a valid frame type')
        files = [f for f in files if frame_type in os.path.basename(f)]
        if tag is not None:
            files = [f for f in files if tag in os.path.basename(f)]
        if target is not None:
            files = [f for f in files if fits.getheader(f)['TARGET']==target]
        if len(files) == 0:
            tag_label = '' if tag is None else ' + ' + tag
            msg = 'No {}{} files found for {} on {}'.\
                format(frame_type, tag_label, df_unit, date)
            logger.critical(msg)
            return None
        else:
            return ImageCollection(files)
