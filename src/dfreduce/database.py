# Standard library
import os
import abc 
from glob import glob

# Third-party
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

# Project
from .log import logger
from .butler import ImageButler, ImageCollection
from .cameras import camera_info as cam_info
from .cameras import get_filter_name
from .flags import DarkFlags, FlatFlags, LightFlags, FlagArray
from . import improc
from . import DFStruct
from . import package_dir
from . import utils 
from . import tasks
default_evening = ['21:00:00', '05:00:00']
default_morning = ['09:00:00', '17:00:00']
flag_dict = dict(dark=DarkFlags, flat=FlatFlags, light=LightFlags)


__all__ = ['DFDatabaseHub', 'DFIndividualFrames', 'DFMasterCals',
           'DFMasterCalsNB']


class DFDatabaseHub(object):
    """
    A convenience class that connects the individual-frame and mastercal 
    databases together. 
    """

    def __init__(self, db_path, data_path, mcals_path, survey = 'UW', **kwargs):
        self.frames = DFIndividualFrames(db_path, data_path, 
                                         survey=survey, **kwargs)
        if survey=='NB':
            self.mcals = DFMasterCalsNB(db_path, mcals_path, 
                                        survey=survey, **kwargs)
        else:
            self.mcals = DFMasterCals(db_path, mcals_path,
                                      survey=survey, **kwargs)
        self.updated_frames_db = False
        self.updated_mcals_db = False
        self.last_resort_mcals_path = os.path.join(mcals_path, 
                                                   'LAST_RESORT_MCALS')

    def _find_last_resort_master_flat(self, frame_id_or_dict):
        """
        If no master flats found for camera, use the last known good master
        flat, which is stored in MasterCals/LAST_RESORT_MCALS
        """
        frame_id = self.frames.check_frame_id(frame_id_or_dict)
        frame = self.frames.df.loc[frame_id]
        path = self.get_last_flat_path_for_frame(frame_id)
        date = pd.to_datetime(fits.getheader(path)['DATE']).date().isoformat()
        results = DFStruct(path=path, frame_id=-99, date=date)
        return results 

    def _find_nearest_mastercal(self, frame_id_or_dict, kind):
        """
        Find the nearest master cal to a given image.

        Parameters
        ----------
        frame_id_or_dict : int or dict
            Unique frame id number or dict with the required params to 
            identify a frame uniquely.
        kind : str
            The kind of master cal -- flat or dark.

        Returns
        -------
        path : str
            Full path to master cal.
        frame_id : int 
            The frame_id of the master cal.
        date : str 
            The date of the master cal.
        """
        frame_id = self.frames.check_frame_id(frame_id_or_dict)
        frame = self.frames.df.loc[frame_id]
        errmsg = f'No master {kind} found for frame {frame_id} '
        errmsg += f'serialno {frame.serialno} on {frame.date}'

        kw = dict(frame_type=kind, serialno=frame.serialno)
        if kind == 'dark':
            kw.update(dict(exptime=frame.exptime))
        if kind == 'flat' and isinstance(self.mcals,DFMasterCalsNB):
            try:
                kw.update(dict(tilt=str(frame.tiltgoal)))
            except ValueError:
                logger.critical(errmsg + ' < No tiltgoal in frame header >')
                return None

        mcals = self.mcals.query('date', **kw)
                              
        mcal_dates = pd.to_datetime(mcals.date)
        frame_date  = pd.to_datetime(frame.date)

        if len(mcal_dates) > 0:
            date_diffs = np.abs(mcal_dates - frame_date)
            nearest_id = date_diffs.idxmin()
            nearest_date = mcal_dates.loc[nearest_id].date()
            msg = 'Nearest master {} for frame=({}, {}, {}, {}) made on {}'
            msg = msg.format(kind, frame.serialno, frame.frame_type, 
                             frame.exptime, frame.date, 
                             nearest_date.isoformat())
            logger.debug(msg)
            path = getattr(self.mcals, f'get_{kind}_path')(nearest_id)
            results = DFStruct(path=path,
                               frame_id=nearest_id, 
                               date=nearest_date.isoformat())
        else:
            logger.critical(errmsg)

            results = None
        return results

    def find_nearest_dark(self, frame_id_or_dict):
        return self._find_nearest_mastercal(frame_id_or_dict, 'dark')

    def find_nearest_flat(self, frame_id_or_dict):
        frame_id = self.frames.check_frame_id(frame_id_or_dict)
        results = self._find_nearest_mastercal(frame_id, 'flat')
        if results is None:
            results = self._find_last_resort_master_flat(frame_id)
            msg = 'Using last resort flat from' 
            logger.warning(f'{msg} {results.date} for frame {frame_id}')
        return results

    def get_last_flat_path_for_frame(self, frame_id_or_dict):
        frame_id = self.frames.check_frame_id(frame_id_or_dict)
        frame = self.frames.df.loc[frame_id]
        path = os.path.join(self.last_resort_mcals_path, 'master_flats')
        path = os.path.join(path, f'master_{frame.serialno}_flat.fits')
        return path

    def _update_nearest_mastercal(self, f_id, kind):
        mcalinfo = getattr(self, f'find_nearest_{kind}')(f_id)
        if mcalinfo is not None:
            kw = {f'master_{kind}_id': mcalinfo['frame_id'],\
                  f'master_{kind}_date': mcalinfo['date']}
            self.frames.update(f_id, **kw)
        self.updated_frames_db = True

    def _update_nearest_mastercal_by_date(self, date, kind, frame_type=None):
        kw = dict(date=date)
        if frame_type is not None:
            kw['frame_type'] = frame_type
        frames_date = self.frames.query(**kw)
        for f_id in frames_date.index:
            self._update_nearest_mastercal(f_id, kind)
        self.updated_frames_db = True

    def update_nearest_master_dark(self, f_id):
        self._update_nearest_mastercal(f_id, 'dark')

    def update_nearest_master_flat(self, f_id):
        self._update_nearest_mastercal(f_id, 'flat')

    def update_nearest_master_darks_by_date(self, date, frame_type=None):
        self._update_nearest_mastercal_by_date(date, 'dark', frame_type)

    def update_nearest_master_flats_by_date(self, date, frame_type=None):
        self._update_nearest_mastercal_by_date(date, 'flat', frame_type)

    def get_frame_mcal_info(self, frame_id_or_dict):
        frame_id = self.frames.check_frame_id(frame_id_or_dict)
        cols = ['master_dark_id', 'master_dark_date', 'master_flat_id', 
                'master_flat_date', 'date', 'frame_type', 
                'exptime', 'serialno']
        info = self.frames.df.loc[frame_id, cols]
        frame_path = self.frames.get_frame_path(frame_id)
        if pd.notna(info.master_dark_id):
            dark_path = self.mcals.get_dark_path(info.master_dark_id)
        else:
            if info.frame_type == 'flat':
                msg = f'Flat {frame_id} has no master dark.'
                logger.debug(msg + ' Looking for single exposure dark.')
                single_dark_paths = self.frames.query_frame_paths(
                    date=info.date, frame_type='dark', serialno=info.serialno,
                    exptime=info.exptime, is_good=True
                )
                if len(single_dark_paths) > 0:
                    logger.warning(f'Using single dark for flat {frame_id}')
                    idx = single_dark_paths.index[0]
                    dark_path = single_dark_paths.loc[idx]
                    info.master_dark_id = -1 * idx
                    info.master_dark_date = info.date
                else:
                    dark_path = None
            else:
                dark_path = None
        if pd.isna(info.master_flat_id):
            flat_path = None
        elif info.master_flat_id >= 0:
            flat_path = self.mcals.get_flat_path(info.master_flat_id)
        else:
            flat_path = self.get_last_flat_path_for_frame(frame_id)
        has_mcals = (flat_path is not None) and (dark_path is not None)
        info = DFStruct(frame_id=frame_id,
                        frame_path=frame_path, 
                        dark_path=dark_path, 
                        flat_path=flat_path, 
                        frame_date=info.date,
                        dark_id=info.master_dark_id,
                        flat_id=info.master_flat_id,
                        dark_date=info.master_dark_date, 
                        flat_date=info.master_flat_date, 
                        has_mcals=has_mcals)
        return info 

    def get_dark_subtracted(self, frame_id_or_dict, **kwargs):
        info = self.get_frame_mcal_info(frame_id_or_dict)
        rm_overscan = True if isinstance(self.mcals,DFMasterCalsNB) else False
        if pd.notna(info.dark_id):
            if info.dark_date != info.frame_date:
                msg = 'Master dark {} date = {} is not the same as frame {} '\
                      'date = {}'
                msg = msg.format(info.dark_id, info.dark_date, 
                                 info.frame_id, info.frame_date) 
                logger.warning(msg)
            image_ds = improc.subtract_master_dark(info.frame_path, 
                                                   info.dark_path, 
                                                   survey=self.frames.survey,
                                                   rm_overscan=rm_overscan,
                                                   **kwargs)
            header = image_ds.header
            header['DARKID'] = info.dark_id
            header['DARKDATE'] = info.dark_date
            frame = DFStruct(info=info, pixels=image_ds.pixels, header=header)
        else:
            msg = 'No master dark found for frame {}'.format(info.frame_id)
            logger.critical(msg)
            frame = None
        return frame
    
    def get_dark_subtracted_and_flat(self, frame_id_or_dict, 
                                     ds_kw={}, ff_kw={}):
        frame = self.get_dark_subtracted(frame_id_or_dict, **ds_kw)
        rm_overscan = True if isinstance(self.mcals,DFMasterCalsNB) else False
        if frame is not None:
            info = frame.info
            if pd.notna(info.flat_id):
                msg = 'Master flat {} date = {} not the same as frame {} '\
                      'date = {}'
                if info.flat_date != info.frame_date:
                    logger.warning(msg.format(info.flat_id, info.flat_date, 
                                              info.frame_id, info.frame_date))
                im_ff = improc.divide_by_master_flat(frame.pixels, 
                                                     info.flat_path, 
                                                     path_or_header=frame.header,
                                                     survey=self.frames.survey,
                                                     rm_overscan=rm_overscan,
                                                     **ff_kw)
                header = im_ff.header
                header['FLATID'] = info.flat_id
                header['FLATDATE'] = info.flat_date
                frame = DFStruct(info=info, pixels=im_ff.pixels, header=header)
            else:
                msg = 'No master flat found for frame {}'.format(info.frame_id)
                logger.critical(msg)
                frame = None
        return frame

    def light_reduction_path_table(self, target):
        paths = self.frames.query_frame_paths(
            where=f"target=='{target}' & frame_type=='light' & flags==0")

        if len(paths) == 0:
            logger.critical('No good lights found')
            return None
           
        index = paths.index
        ids = self.frames.loc[index, ['master_dark_id', 'master_flat_id']]
        isna = pd.isna(ids.master_dark_id) | pd.isna(ids.master_flat_id)
        if isna.sum() == len(index):
            logger.critical('No lights have *both* master darks & flats')
            return None

        paths = paths[~isna]
        index = index[~isna]
        dark_index = ids.master_dark_id[~isna].values
        flat_index = ids.master_flat_id[~isna].values

        flat_paths = []
        for _i, frame_id in zip(flat_index, index.values):
            if _i < 0:
                # last resort flats 
                _p = self.get_last_flat_path_for_frame(frame_id)
            else:
                _p = self.mcals.flat_paths.loc[_i]
            flat_paths.append(_p)

        dark_paths = self.mcals.dark_paths.loc[dark_index]
        extra = self.frames.query('serialno,expnum', index=index)
        filters = [get_filter_name(fn) for fn in paths.values]
        paths = dict(light=paths.values, 
                     master_dark=dark_paths.values, 
                     master_flat=flat_paths, 
                     serialno=extra.serialno.values,
                     filter_name=filters,
                     expnum=extra.expnum.values)
        path_table = pd.DataFrame(paths, index=index)
        path_table.index.name = 'light_id'
        if isna.sum() > 0:
            msg = f'{isna.sum()} lights are missing a master dark and/or flat'
            logger.warning(msg)
        return path_table

    def update_frames(self, frame_id_or_dict, **kwargs):
        self.frames.update(frame_id_or_dict, **kwargs)
        self.updated_frames_db = True

    def add_night_to_mcals(self, date):
        self.mcals.add_night(date)
        self.updated_mcals_db = True

    def write_updates(self, overwrite=False):
        if self.updated_frames_db:
            self.frames.write_database(overwrite)
            self.updated_frames_db = False
        if self.updated_mcals_db:
            self.mcals.write_database(overwrite)
            self.updated_mcals_db = False

    def make_master_darks_by_date(self, date, **kwargs):
        self.mcals.setup_date_paths(date)
        rm_overscan = True if isinstance(self.mcals,DFMasterCalsNB) else False
        darks = self.frames.query(frame_type='dark', date=date, **kwargs)
        darks_by_sn_time = darks.groupby(['serialno', 'exptime'])
        for (serialno, time), frame_ids in darks_by_sn_time.groups.items():
            logger.debug('Sending the following frames to create_master_dark:')
            logger.debug(np.transpose([
                self.frames.query_frame_paths(index=frame_ids).values,\
                darks['exptime'].loc[frame_ids].values]))
            # logger.debug(darks[['files','exptime']].loc[frame_ids])
            darkim_paths = self.frames.query_frame_paths(index=frame_ids).values
            # darkim_paths = darks['files'].loc[frame_ids].values
            tasks.darks.create_master_dark(darkim_paths, serialno=serialno,
                save_path=os.path.join(self.mcals.dark_path,date),
                survey=self.frames.survey, rm_overscan=rm_overscan)

    def _make_master_flats_by_date(self, date):
        self.mcals.setup_date_paths(date)
        flats = self.frames.query(frame_type='flat', date=date, is_good=True)
        flats_by_sn = flats.groupby('serialno')
        for serialno, frame_ids in flats_by_sn.groups.items():
            tasks.flats.create_master_flat(frame_ids, self,
                save_path=os.path.join(self.mcals.flat_path,date))

    def _make_master_flats_by_date_nb(self, date):
        self.mcals.setup_date_paths(date)
        flats = self.frames.query(frame_type='flat', date=date)
        flats_by_sn_tilt = flats.groupby(['serialno', 'tiltgoal'])
        for (serialno, tilt), frame_ids in flats_by_sn_tilt.groups.items():
            logger.debug('Sending the following frames to create_master_flat:')
            logger.debug(np.transpose([
                self.frames.query_frame_paths(index=frame_ids).values,
                flats['tiltgoal'].loc[frame_ids].values]))
            # logger.debug(flats[['files','tiltgoal']].loc[frame_ids])
            tasks.flats.create_master_flat(frame_ids, self, min_num_flats=5,
                tilt=tilt, save_path=os.path.join(self.mcals.flat_path,date),
                rm_overscan=True,survey=self.frames.survey)

    def make_master_flats_by_date(self, date):
        if isinstance(self.mcals,DFMasterCalsNB):
            self._make_master_flats_by_date_nb(date)
        else:
            self._make_master_flats_by_date(date)


class DFDatabaseManager(metaclass=abc.ABCMeta):
    """ 
    Manage and interface with the Dragonfly databases.

    Parameters
    -----------
    db_path : str
        Full path to the directory where the Dragonfly databases are stored.
    db_label : str (optional)
        Unique label for the database file name.
    survey : str (optional)
        The name of the Dragonfly survey in question. This sets the selection
        of (and mapping between) available Dragonfly units and
        camera serial numbers. Default is 'UW' (Dragonfly UltraWide Survey).
    """

    dtype = None 

    def __init__(self, db_path, db_label=None, survey='UW'):
        label = '' if db_label is None else '_' + db_label
        fn = f'df_database_{self.db_type}{label}_{survey.lower()}.csv'
        assert os.path.isdir(db_path), f'{db_path} does not exist'
                                    
        self.df = None
        self.survey = survey.upper()
        self.db_path = db_path
        self.backup_file_name = None
        self.file_name = os.path.join(db_path, fn)
        self.df_unit_names = list(cam_info[survey]['serialno_dict'].keys())

    @property
    def loc(self):
        return self.df.loc

    @property
    def shape(self):
        return self.df.shape

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    @property
    def next_index(self):
        idx = 0 if len(self.df) == 0 else self.df.index[-1] + 1
        return idx

    @property
    def has_been_updated(self):
        if self.backup_file_name is None:
            logger.error('You have to save a backup file to make this check')
            updated = None
        else:
            backup = pd.read_csv(self.backup_file_name, 
                                 index_col='frame_id', 
                                 dtype=self.dtype)
            updated = not backup.equals(self.df)
        return updated

    @abc.abstractmethod 
    def add_night(self, date):
        raise NotImplementedError()

    def check_frame_id(self, frame_id_or_dict):
        _type = type(frame_id_or_dict)
        if _type == dict:
            frame_id = self.get_frame_id(frame_id_or_dict)
        elif _type == int or _type == np.int64:
            frame_id = frame_id_or_dict
        else:
            msg = '{} is not a valid frame identifier'.format(frame_id_or_dict)
            raise Exception(msg + ' --> frame_id_or_dict must be int or dict.')
        return frame_id

    def select_fields(self, select='*'):
        """Returns user-specified fields to view from the database."""
        if select == '*' or select == 'all':
            selection = self.fields
        elif select == 'id':
            selection = self.id_fields
        elif select == 'min' or select == 'minimal':
            selection = self.minimal_fields
        else:
            if type(select) == str:
                selection = select.replace(' ', '').split(',')
            elif type(select) == list:
                selection = select
            else:
                Exception('{} is not a valid select option'.format(select))
            for sel in selection:
                assert sel in self.fields, '{}: not a valid field'.format(sel)
        return selection

    def preview(self, select='*', where=None, index=None, n=4, **kwargs):
        """Preview the first n rows the database."""
        results = self.query(select, where, index, **kwargs)
        return results.head(n=n)

    def show(self, select='*'):
        """Show the full data table."""
        selected_fields = self.select_fields(select)
        return self.df[selected_fields]

    def database_exists(self):
        """Check to see whether or not the database exists."""
        return os.path.isfile(self.file_name)

    def create_empty_database_df(self, force=False):
        """ Create an empty DataFrame"""
        if self.df is None or force:
            logger.debug('Creating empty DataFrame')
            self.df = pd.DataFrame(columns=self.fields)
            self.df.index.name = 'frame_id'
        else:
            msg = 'A database DataFrame already exists. Use force = True '
            logger.error(msg + 'if this is really what you want')

    def load_database(self, save_backup_file=True):
        """ 
        Load the database.

        Parameters
        ----------
        save_backup_file :  bool (optional)
            If True, save a back up copy of the database file (if it exists)
            in the same directory as a hidden file.
        """
        name = os.path.basename(self.file_name)[:-4]
        if not self.database_exists():
            check_str = 'DB {} does not exist - CREATING empty DB DataFrame' 
            logger.info(check_str.format(name))
            self.create_empty_database_df()
        else:
            logger.debug('DB {} exists - LOADING'.format(name))
            self.df = pd.read_csv(self.file_name, 
                                  index_col='frame_id', 
                                  dtype=self.dtype)
            base_fn = os.path.basename(self.file_name)
            bckup_fn = os.path.dirname(self.file_name)
            bckup_fn = os.path.join(self.db_path, '.bckup_' + base_fn)
            if save_backup_file:
                logger.debug('Writing backup database to ' + bckup_fn)
                self.backup_file_name = bckup_fn
                self.df.to_csv(bckup_fn, index=True)

    def update(self, frame_id_or_dict, **kwargs):
        """ 
        Update parameters in DB DataFrame for individual or multiple frames. 
        This will NOT write the updated database to its associated csv file. 
        If you want to save the updates use the write_database method.

        Parameters
        ----------
        frame_id_or_dict : int or dict or list-like
            Unique frame id number or dict with the required params to 
            identify a frame uniquely.
        **kwargs 
            Parameter(s) as the keyword(s) with the update as the value(s). 
            The input options are flexible. See examples.
            
        Examples
        --------
        - db.update(666, is_good=True, flags=0, expnum=101)
        - db.update([2, 501, 3005], is_good=[True, False, True])

        NOTES
        -----
        - To reset a field, set it to pd.NA.
        """
        _id = frame_id_or_dict
        assert len(kwargs) > 0, 'You did not pass anything to update!'
        keys = list(kwargs.keys())
        vals = list(kwargs.values())

        if utils.is_list_like(_id):
            assert len(keys) == len(vals), 'key and val must have same length'
            for v in vals:
                assert utils.is_list_like(v), 'all values must be list-like!'
                check = len(v) == len(_id)
                assert check, '# of updates value must match # of frames!'
            id_list = []
            for id_or_dict in _id:
                id_list.append(self.check_frame_id(id_or_dict))
        else:
            id_list = [self.check_frame_id(frame_id_or_dict)]

        for k, v in zip(keys, vals):
            if k not in self.df.keys():
                check_str = "Property {} does not exist in database type {}"
                raise Exception(check_str.format(k, self.db_type))
            if k != 'flags':
                self.df.loc[id_list, k] = v
            else:
                v = v if utils.is_list_like(v) else [v]
                v = np.asarray(v)
                isna = self.df.loc[id_list, k].isna()
                notna = self.df.loc[id_list, k].notna()
                if notna.sum() > 0:
                    new_val_notna = pd.notna(v)
                    update_notna = notna.values & new_val_notna
                    v_notna = v[update_notna]
                    if update_notna.sum() > 0:
                        is_pos = v_notna > 0
                        is_neg = v_notna < 0
                        if notna.loc[is_pos].sum() > 0:
                            idx = notna.loc[is_pos].index
                            self.df.loc[idx, k] |= v_notna[is_pos]
                        if notna.loc[is_neg].sum() > 0:
                            idx = notna.loc[is_neg].index
                            abs_vals = np.abs(v_notna[is_neg])
                            self.df.loc[idx, k] &= ~abs_vals
                    update_na = notna.values & (~new_val_notna)
                    if update_na.sum() > 0:
                        idx = notna.loc[update_na].index
                        self.df.loc[idx, k] = v[update_na]
                if isna.sum() > 0:
                    idx = isna.loc[isna.values].index
                    self.df.loc[idx, k] = v[isna.values]

    def write_database(self, overwrite=False):
        """
        Write database to a csv; but check if it exists first

        Parameters
        ----------
        overwrite : bool (optional)
            If True, database changes will be permanently written out.
            Default is False.
        """
        if not os.path.isfile(self.file_name):
            logger.info('Creating database csv file ' + self.file_name)
            self.df.to_csv(self.file_name, index=True)
        elif overwrite:
            logger.debug('Overwriting database: ' + self.file_name)
            self.df.to_csv(self.file_name, index=True)
        else:
            logger.error('Database exists -- use overwrite = True to save')

    def mask_database(self, **kwargs):
        """ 
        Create a mask for the table based on a set of fields.

        EXAMPLES
        --------
        my_mask = self.mask_database(date='2019-11-20', serialno='T13090564')
        """
        keys = list(kwargs.keys())
        notna = self.df[keys].notna().sum(axis=1) == len(keys)
        mask = np.all(self.df[keys] == pd.Series(kwargs), axis=1)
        mask[~notna] = False
        return mask.values

    def get_frame_id(self, frame_dict={}, **kwargs):
        id_fields = self.id_fields
        frame_dict.update(kwargs)
        for param in id_fields:
            if param not in frame_dict.keys():
                raise Exception('Must give {} to ID a row'.format(id_fields))
        row = self.query(select='serialno', **frame_dict)
        assert len(row) == 1, 'Something is wrong with the DB or frame_dict!?!'
        frame_id = row.index[0]
        return frame_id

    def query(self, select='*', where=None, index=None, **kwargs):
        """ 
        Grab a selection of frames from the database. 

        Parameters
        ----------
        select : str or list
            Parameters to select from the database. 
            e.g., '*', ['frame_type', 'exptime'], or 'frame_type,exptime'
        where : str (optional)
            sql-like condition 
            e.g., 'flags > 5 & df_unit == 201'
        index : pd.Int64Index
            Database table frame_id index 
        **kwargs
            Conditions passed as keyword arguments.
            e.g., query('*', frame_type='dark', exptime=600.0)

        Returns
        -------
        results : pd.DataFrame
            The result from the database query.

        Notes
        -----
        It is only possible to use a single condition method at a time. 
        """
        selected_fields = self.select_fields(select)
        if index is not None:
            results = self.df.loc[index, selected_fields]
        elif where is not None:
            astype = {}
            for nullable in self.nullable_ints:
                if nullable in where:
                    if self.df[nullable].sum() > 0:
                        astype[nullable] = float
                    else:
                        msg = f'The {nullable} column is empty --> ' 
                        msg += 'SQL-like queries will not work'
                        logger.error(msg)
                        return None
            results = self.df.astype(astype).query(where)[selected_fields]
            for nullable in self.nullable_ints:
                if nullable in selected_fields:
                    results[nullable] = results[nullable].astype('Int64')
        elif len(kwargs) > 0:
            mask = self.mask_database(**kwargs)
            results = self.df.loc[mask, selected_fields]
        else:
            results = self.df[selected_fields]
        return results

    def query_index(self, where=None, **kwargs):
        if where is not None:
            index = self.query('frame_type', where=where).index
        elif len(kwargs) > 0:
            mask = self.mask_database(**kwargs)
            index = self.df.index[mask]
        else:
            index = self.df.index
        return index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        return self.df[key]


class DFIndividualFrames(DFDatabaseManager):
    """ Dragonfly database for individual (raw) frames """

    db_type = 'individual_frames'

    fields = ['serialno', 'df_unit', 'date', 'frame_type', 'expnum', 'target',
              'exptime', 'filter_name', 'ra', 'dec', 'alt', 'az', 'ccd_temp', 
              'hdr_datetime', 'master_dark_date', 'master_dark_id', 
              'master_flat_date', 'master_flat_id', 'flags', 'is_good']
    nullable_ints = ['master_dark_id', 'master_flat_id', 'flags']

    id_fields = ['serialno', 'date', 'frame_type', 'expnum']

    surveys = ['UW', 'NGC', 'NB', 'LE', 'DD']

    dtype = dict(
        serialno=str, df_unit=int, date=str, frame_type=str, target=str, 
        expnum=int, filter_name=str, ra=str, dec=str, alt=str, az=str, 
        ccd_temp=str, hdr_datetime=str, master_dark_date=pd.StringDtype(),
        master_dark_id=pd.Int64Dtype(), master_flat_date=pd.StringDtype(), 
        master_flat_id=pd.Int64Dtype(), flags=pd.Int64Dtype(), 
        is_good='boolean'
    )

    def __init__(self, db_path, data_path, db_label=None, survey='UW', 
                 save_backup_file=True):

        assert os.path.isdir(data_path), f'{data_path} does not exist'
        utils.mkdir_if_needed(db_path)
        super(DFIndividualFrames, self).__init__(db_path, db_label, survey)

        self.data_path = data_path
        self.minimal_fields = ['serialno', 'df_unit', 'date', 'frame_type',
                               'expnum', 'target', 'exptime', 
                               'master_dark_id', 'master_flat_id', 
                               'flags', 'is_good']
        if (self.survey =='NB') and ('tilt' not in self.fields):
            self.fields += ['tilt','rawtilt','corrtilt','tiltgoal']
        self.load_database(save_backup_file=save_backup_file)
        self._sc = None

    @property
    def targets(self):
        unique_targets = np.array([
            t for t in np.unique(self.df.target)\
                    if 'dark' not in t.lower()\
                    if 'flat' not in t.lower()\
                    if 'unknown' not in t.lower()
        ])
        return unique_targets

    @property
    def frame_relpaths(self):
        df_units = 'Dragonfly' + self.df.df_unit.astype(str)
        paths = df_units.str.cat(self.df.date, sep='/').\
                         str.cat(self.df.serialno, sep='/').\
                         str.cat(self.df.expnum.astype(str), sep='_').\
                         str.cat(self.df.frame_type, sep='_') + '.fits'
        return paths

    @property
    def frame_paths(self):
        paths = self.frame_relpaths
        paths = paths.apply(lambda p: os.path.join(self.data_path, p))
        return paths

    @property
    def skycoords(self):
        if self._sc is not None:
            # only regenerate if the table has been updated
            if len(self._sc) != len(self.df):
                self._sc = SkyCoord(self.df.ra, 
                                    self.df.dec.str.replace('d', ':'), 
                                    unit=(u.hourangle, u.deg))
        else:
            self._sc = SkyCoord(self.df.ra, 
                                self.df.dec.str.replace('d', ':'), 
                                unit=(u.hourangle, u.deg))
        return self._sc

    def _check_mcals_path(self, mcals_db_or_dbpath):
        if type(mcals_db_or_dbpath) == str:
            mcals_db = DFMasterCals(self.db_path, mcals_db_or_dbpath)
        else:
            mcals_db = mcals_db_or_dbpath
        return mcals_db

    def get_frame_path(self, frame_id_or_dict, basename=False):
        frame_id = self.check_frame_id(frame_id_or_dict)
        path = os.path.join(self.data_path, self.frame_relpaths.loc[frame_id])
        if basename:
            path = os.path.basename(path)
        return path

    def get_frame_header(self, frame_id_or_dict):
        path = self.get_frame_path(frame_id_or_dict)
        header = fits.getheader(path)
        return header

    def get_frame_shape(self, frame_id_or_dict):
        header = self.get_frame_header(frame_id_or_dict)
        shape = (header['NAXIS2'], header['NAXIS1'])
        return shape

    def get_frame_part_of_day(self, frame_id_or_dict, evening=default_evening,
                              morning=default_morning):
        frame_id = self.check_frame_id(frame_id_or_dict)
        obs_date = self.df.loc[frame_id, 'date'] + 'T'

        evening_start_time = pd.Timestamp(obs_date + evening[0])
        evening_end_time = pd.Timestamp(obs_date + evening[1])
        evening_end_time += pd.Timedelta('1 days')

        morning_start_time = pd.Timestamp(obs_date + morning[0])
        morning_start_time += pd.Timedelta('1 days')
        morning_end_time = pd.Timestamp(obs_date + morning[1])
        morning_end_time += pd.Timedelta('1 days')

        datetime = pd.to_datetime(self.df.loc[frame_id, 'hdr_datetime'])
        if datetime >= evening_start_time  and datetime <= evening_end_time:
            part_of_day = 'evening'
        elif datetime >= morning_start_time and datetime <= morning_end_time:
            part_of_day = 'morning'
        else:
            part_of_day = 'unknown'

        return part_of_day

    def query_part_of_day(self, index=None, evening=default_evening, 
                          morning=default_morning, **kwargs):

        selection = self.query('date,hdr_datetime', index=index, **kwargs)
        obs_date = selection['date'] + 'T'

        evening_start_time = pd.to_datetime(obs_date + evening[0])
        evening_end_time = pd.to_datetime(obs_date + evening[1])
        evening_end_time += pd.Timedelta('1 days')

        morning_start_time = pd.to_datetime(obs_date + morning[0])
        morning_start_time += pd.Timedelta('1 days')
        morning_end_time = pd.to_datetime(obs_date + morning[1])
        morning_end_time += pd.Timedelta('1 days')

        datetime = pd.to_datetime(selection['hdr_datetime'])

        part_of_day = pd.Series(np.repeat('unknown', len(selection)), 
                                index=selection.index, name='part_of_day')

        evening_mask = datetime >= evening_start_time
        evening_mask &= datetime <= evening_end_time
        part_of_day[evening_mask] = 'evening'

        morning_mask = datetime >= morning_start_time
        morning_mask &= datetime <= morning_end_time
        part_of_day[morning_mask] = 'morning'

        return part_of_day

    def query_frame_paths(self, where=None, index=None, 
                          full_paths=True, **kwargs):
        """ 
        Returns an array of paths to images satisfying the input conditions.
        """
        if index is None:
            index = self.query_index(where, **kwargs)
        results = self.frame_relpaths.loc[index]
        if full_paths:  
            results = results.apply(lambda p: os.path.join(self.data_path, p))
        return results

    def query_skycoords(self, where=None, **kwargs):
        index = self.query_index(where, **kwargs)
        return self.skycoords[index.values]

    def query_flag_strings(self, where=None, joiner=',', lower=False, 
                           **kwargs):
        frames = self.query('frame_type,flags', where=where, **kwargs)
        kw = dict(joiner=joiner, lower=lower)
        flags = frames.apply(
            lambda x: flag_dict[x.frame_type](x.flags).to_string(**kw)\
                      if pd.notna(x.flags) else x.flags, axis=1
        )
        return flags

    def query_flag_array(self, frame_type, where=None, **kwargs):
        frames = self.query('flags', frame_type=frame_type, 
                            where=where, **kwargs)
        return FlagArray(frames.flags, frame_type)

    def add_night(self, date, df_unit=None):
        """
        Add observations from a given night to the database table. 
        This will NOT write the updated database to its associated csv file. 
        If you want to save the updates use the write_database method.

        Parameters
        ----------
        date : str
            Add frames from this date.
        df_unit : int or list
            Only search for data for these df_unit numbers. If None, 
            search for all df_unit numbers.

        NOTES
        -----
        Rows only added if the frame does *not* yet exist
        in the database. To make any adjustments, use the
        update() method.
        """
        butler = ImageButler(base_path=self.data_path,survey=self.survey)

        new_index = pd.Index(data=[], name='frame_id')

        if df_unit is not None:
            unit_names = df_unit if utils.is_list_like(df_unit) else [df_unit]
            unit_names = [f'Dragonfly{n}' for n in unit_names]
        else:
            unit_names = self.df_unit_names

        for dragonfly in unit_names:
            df_unit = int(dragonfly[-3:])
            serialno = cam_info[self.survey]['serialno_dict'][dragonfly]

            all_files = butler.df_unit_files(df_unit, date)
            num_existing_files = len(all_files)

            if num_existing_files > 0:

                img_col = ImageCollection(all_files, self.survey)
                if len(img_col.corrupt_files) > 0:
                    num_corrupt = len(img_col.corrupt_files)
                    msg = '{} corrupt files on {} for Dragonfly{}' 
                    logger.warning(msg.format(num_corrupt, date, df_unit))

                num_existing_files -= len(img_col.corrupt_files)
                frames_in_table = self.frame_relpaths.values
                base_fnames = np.array([os.path.relpath(f, self.data_path)
                                        for f in img_col.files])

                # no duplicates mask
                no_dups = ~np.in1d(base_fnames, frames_in_table)
                no_dups = np.array(no_dups)
    
                if no_dups.sum() == 0:
                    msg = 'All {} frames on {} for Dragonfly{} '
                    msg = msg.format(num_existing_files, date, df_unit)
                    logger.warning(msg + 'already on database')
                    continue

                base_fnames = base_fnames[no_dups] 
                num_new_files = len(base_fnames)
                frame_types = [s.split('_')[-1].split('.fits')[0]
                               for s in base_fnames]

                if img_col.num_exo_files_removed > 0:
                    msg = f'Removed {img_col.num_exo_files_removed} '
                    logger.warning(msg + 'exoplanet files.')
                num_files = num_existing_files - img_col.num_exo_files_removed
                if num_new_files < num_files:
                    msg = 'Skipped {} files for {} on {} to avoid duplicates'
                    file_diff = num_files - num_new_files 
                    logger.warning(msg.format(file_diff, dragonfly, date))

                # nullable int array
                na_list = [pd.NA] * num_new_files
                nullable_int_arr = pd.array(na_list, dtype='Int64')
                df_unit_arr = np.repeat(df_unit, num_new_files)

                data = {'serialno': [serialno] * num_new_files,
                        'df_unit': df_unit_arr.astype(int),
                        'date': [date] * num_new_files,
                        'frame_type': frame_types,
                        'expnum': img_col.expnum[no_dups],
                        'target': img_col.targets[no_dups],
                        'exptime': img_col.exptimes[no_dups],
                        'filter_name': img_col.filters[no_dups],
                        'is_good': na_list,
                        'flags': nullable_int_arr,
                        'master_dark_id': nullable_int_arr,
                        'master_dark_date': na_list,
                        'master_flat_id': nullable_int_arr,
                        'master_flat_date': na_list,
                        'ra': img_col.ra[no_dups],
                        'dec': img_col.dec[no_dups],
                        'alt': img_col.alt[no_dups],
                        'az': img_col.az[no_dups],
                        'ccd_temp': img_col.ccd_temp[no_dups],
                        'hdr_datetime': img_col.datetime[no_dups]}

                if (self.survey =='NB'):
                    data['tilt'] = img_col.tilt[no_dups]
                    data['rawtilt'] = img_col.rawtilt[no_dups]
                    data['corrtilt'] = img_col.corrtilt[no_dups]
                    data['tiltgoal'] = img_col.tiltgoal[no_dups]

                index = pd.RangeIndex(self.next_index, 
                                      self.next_index + num_new_files, 
                                      name='frame_id')
                new_index = new_index.union(index)
                new_rows = pd.DataFrame(data, index=index,
                                        columns=list(self.df.keys()))
                self.df = self.df.append(new_rows)

            else:
                logger.warning('No files found for {} on {}'.\
                               format(dragonfly, date))

        return new_index


class DFMasterCals(DFDatabaseManager):
    """Dragonfly Database for master calibration frames"""

    db_type = 'master_cals'
    fields = ['serialno', 'date', 'frame_type', 'exptime', 'exists']
    id_fields =  ['serialno', 'date', 'frame_type', 'exptime']
    dtype = dict(serialno=str, date=str, frame_type=str, 
                 exptime=float, exists='boolean')

    def __init__(self, db_path, mcals_path, db_label=None, survey='UW', 
                 save_backup_file=True):

        super(DFMasterCals, self).__init__(db_path, db_label, survey)
        self.mcals_path = mcals_path
        utils.mkdir_if_needed(mcals_path)
        self.dark_path = os.path.join(self.mcals_path, 'master_darks')
        utils.mkdir_if_needed(self.dark_path)
        self.flat_path = os.path.join(self.mcals_path, 'master_flats')
        utils.mkdir_if_needed(self.flat_path)

        self.minimal_fields = ['serialno', 'date', 'exptime']

        self.load_database(save_backup_file=save_backup_file)

    @property
    def dark_paths(self):
        darks = self.query('*', frame_type='dark')
        if len(darks) > 0:
            paths = darks.date.str.cat(darks.serialno, '/master_').\
                               str.cat(darks.frame_type, '_').\
                               str.cat(darks.exptime.astype(str), '_')
            paths = paths + '.fits'
            paths = paths.apply(lambda f: os.path.join(self.dark_path, f))
        else:
            logger.warning('No master darks in database!')
            paths = None
        return paths

    @property
    def flat_paths(self):
        flats = self.query('*', frame_type='flat')
        if len(flats) > 0:
            paths = flats.date.str.cat(flats.serialno, '/master_').\
                               str.cat(flats.frame_type, '_') + '.fits'
            paths = paths.apply(lambda f: os.path.join(self.flat_path, f))
        else:
            logger.warning('No master flats in database!')
            paths = None
        return paths

    def get_dark_path(self, frame_id_or_dict):
        frame_id = self.check_frame_id(frame_id_or_dict)
        path = self.dark_paths.loc[frame_id]
        return path

    def get_flat_path(self, frame_id_or_dict):
        frame_id = self.check_frame_id(frame_id_or_dict)
        if frame_id < 0:
            raise Exception('frame_id < 0: Did you want a last resort flat?')
        path = self.flat_paths.loc[frame_id]
        return path 

    def setup_date_paths(self, date):
        logger.debug('Setting up master cal date paths')
        date_path = os.path.join(self.dark_path, date)
        utils.mkdir_if_needed(date_path)
        date_path = os.path.join(self.flat_path, date)
        utils.mkdir_if_needed(date_path)

    def build_dark_path(self, date, serialno, exptime):
        fn = 'master_{}_dark_{:.1f}.fits'.format(serialno, exptime)
        date_path = os.path.join(self.dark_path, date)
        if not os.path.isdir(date_path):
            logger.warning(date_path + ' does not exist.')
        return os.path.join(date_path, fn)

    def build_flat_path(self, date, serialno, part_of_day=None):
        part_of_day = '' if part_of_day is None else '_' + part_of_day.lower()
        fn = 'master_{}_flat{}.fits'.format(serialno, part_of_day)
        date_path = os.path.join(self.flat_path, date)
        if not os.path.isdir(date_path):
            logger.warning(date_path + ' does not exist.')
        return os.path.join(date_path, fn) 

    def add_night(self, date, which='both'):
        """ 
        Add mastercals from a given night's directory to database table. 
        This will NOT write the updated database to its associated csv file. 
        If you want to save the updates use the write_database method.

        Parameters
        ----------
        date : str
            Add frames from this date.

        NOTES
        -----
        Rows only added if the frame does *not* yet exist
        in the database. To make any adjustments, use the
        update() method.
        """
        butler = ImageButler(base_path=self.mcals_path,survey=self.survey)

        fnames = [] 
        if which=='dark' or which=='darks' or which=='both':
            mdark_path = os.path.join('master_darks', date) 
            darks = butler.glob_files(mdark_path, tag='dark')
            fnames.extend([os.path.basename(fn) for fn in darks])
        if which=='flat' or which=='flats' or which=='both':
            mflat_path = os.path.join('master_flats', date) 
            flats = butler.glob_files(mflat_path, tag='flat')
            flat_base = [os.path.basename(fn) for fn in flats\
                                              if 'morning' not in fn\
                                              if 'evening' not in fn\
                                              if fn[-9:] == 'flat.fits']
            fnames.extend(flat_base)
        num_existing_files = len(fnames)

        if num_existing_files > 0:
            kw = dict(date=date) 
            if which != 'both':
                kw.update(dict(frame_type=which.replace('s', '')))
            _tab = self.query(select='serialno,frame_type,exptime', **kw)
            _tab.fillna('flat-no-exptime', inplace=True)
            prev_ids = _tab.serialno.str.cat(_tab.frame_type, sep='-').\
                                     str.cat(_tab.exptime.astype(str), sep='-')
            new_ids = []
            serialnos = []
            frame_types = []
            exptimes = []
            for fn in fnames:
                sn = fn.split('_')[1]
                ftype = fn.split('_')[2].replace('.fits', '')
                serialnos.append(sn)
                frame_types.append(ftype)
                if 'dark' in fn:
                    t = float(fn.split('_')[-1][:-5])
                    exptimes.append(t)
                elif 'flat' in fn:
                    t = 'flat-no-exptime'
                    exptimes.append(pd.NA)
                else:
                    raise Exception('Unrecognized MasterCal file --> ' + fn)
                new_ids.append('{}-{}-{}'.format(sn, ftype, t))
            serialnos = np.array(serialnos)
            frame_types = np.array(frame_types)
            exptimes = np.array(exptimes)
            
            # no duplicate mask
            already_in_db = np.in1d(new_ids, prev_ids.values)
            num_new_files = (~already_in_db).sum()
            idx_found = prev_ids.index[np.in1d(prev_ids.values, new_ids)]
            mcal_exists = [True] * len(idx_found)

            if num_new_files == 0:
                msg = 'All {} master cals on {} already in database '
                msg += '--> setting exits = True'
                logger.warning(msg.format(num_existing_files, date))
                self.update(idx_found, exists=mcal_exists)
                return None

            elif num_new_files < num_existing_files:
                warning_str = 'Skipped {} files on {} to avoid duplicates'
                file_diff = num_existing_files - num_new_files
                self.update(idx_found, exists=mcal_exists)
                logger.warning(warning_str.format(file_diff, date))

            new_mask = ~already_in_db
            mcal_exists = [True] * len(serialnos[new_mask])
            new_rows = pd.DataFrame(dict(serialno=serialnos[new_mask], 
                                         frame_type=frame_types[new_mask], 
                                         exptime=exptimes[new_mask],
                                         exists=mcal_exists))

            new_rows.reset_index(drop=True, inplace=True)
            index = pd.RangeIndex(self.next_index, 
                                  self.next_index + len(new_rows), 
                                  name='frame_id')
            new_rows.set_index(index, inplace=True)
            new_rows['date'] = date
            self.df = self.df.append(new_rows, sort=False)
            msg = 'Added {} mastercals to DB'.format(len(new_rows))
            logger.info(msg)

        else:
            logger.warning('No new files found for {}'.format(date))
            new_rows = None

        return new_rows


class DFMasterCalsNB(DFMasterCals):
    """Dragonfly Database for NB master calibration frames"""

    db_type = 'master_cals'
    fields = ['serialno', 'date', 'frame_type', 'exptime', 'tilt']
    id_fields =  ['serialno', 'date', 'frame_type', 'exptime', 'tilt']
    nullable_ints = []

    def __init__(self, db_path, mcals_path, db_label=None, survey='UW',
                 save_backup_file=True):

        super(DFMasterCalsNB, self).__init__(db_path, mcals_path, db_label,
                                             survey, save_backup_file)

        self.minimal_fields = ['serialno', 'date', 'exptime', 'tilt']

    @property
    def dark_paths(self):
        darks = self.query('*', frame_type='dark')
        if len(darks) > 0:
            paths = darks.date.str.cat(darks.serialno, '/master_').\
                               str.cat(darks.frame_type, '_').\
                               str.cat(darks.exptime.astype(str),'_') + '.fits'
            paths = paths.apply(lambda f: os.path.join(self.dark_path, f))
            paths = paths.apply(lambda f: f if os.path.isfile(f) else \
                               f.replace('.0',''))
        else:
            logger.warning('No master darks in database!')
            paths = None
        return paths

    @property
    def flat_paths(self):
        flats = self.query('*', frame_type='flat')
        if len(flats) > 0:
            paths = flats.date.str.cat(flats.serialno, '/master_').\
                               str.cat(flats.frame_type, '_').\
                               str.cat(flats.tilt.astype(str),'_') + '.fits'
            paths = paths.apply(lambda f: os.path.join(self.flat_path, f))
            paths = paths.apply(lambda f: f if os.path.isfile(f) else \
                                f.replace('.0',''))
        else:
            logger.warning('No master flats in database!')
            paths = None
        return paths

    def add_night(self, date):
        """
        Add mastercals from a given night's directory to database table. 
        This will NOT write the updated database to its associated csv file. 
        If you want to save the updates use the write_database method.

        Parameters
        ----------
        date : str
            Add frames from this date.

        NOTES
        -----
        Rows only added if the frame does *not* yet exist
        in the database. To make any adjustments, use the
        update() method.
        """
        butler = ImageButler(base_path=self.mcals_path,survey=self.survey)

        mdark_path = os.path.join('master_darks', date)
        darks = butler.glob_files(mdark_path, tag='dark')
        mflat_path = os.path.join('master_flats', date)
        flats = butler.glob_files(mflat_path, tag='flat')
        fnames = [os.path.basename(fn) for fn in darks]
        flat_base = [os.path.basename(fn) for fn in flats\
                                          if 'morning' not in fn\
                                          if 'evening' not in fn]
        fnames.extend(flat_base)
        num_existing_files = len(fnames)
        new_rows = None

        if num_existing_files > 0:
            _tab = self.query(select='serialno,frame_type,exptime', date=date)
            _tab.fillna('flat-no-exptime', inplace=True)
            _tabtilt = self.query(select='tilt', date=date)
            _tabtilt.fillna('dark-no-tilt', inplace=True)
            _tab = pd.concat([_tab, _tabtilt], axis=1, sort=False)

            prev_ids = _tab.serialno.str.cat(_tab.frame_type, sep='-').\
                                     str.cat(_tab.exptime.astype(str), sep='-').\
                                     str.cat(_tab.tilt.astype(str), sep='-')
            new_ids = []
            serialnos = []
            frame_types = []
            exptimes = []
            tilts = []
            for fn in fnames:
                sn = fn.split('_')[1]
                ftype = fn.split('_')[2].replace('.fits', '')
                serialnos.append(sn)
                frame_types.append(ftype)
                if 'dark' in fn:
                    time = float(fn.split('_')[-1][:-5])
                    exptimes.append(time)
                    tilt = 'dark-no-tilt'
                    tilts.append(pd.NA)
                elif 'flat' in fn:
                    time = 'flat-no-exptime'
                    exptimes.append(pd.NA)
                    tilt = fn.split('_')[-1][:-5]
                    tilts.append(tilt)
                else:
                    raise Exception('Unrecognized MasterCal file --> ' + fn)
                new_ids.append('{}-{}-{}-{}'.format(sn, ftype, time, tilt))

            # no duplicate mask
            no_dup = ~np.in1d(new_ids, prev_ids.values)
            num_new_files = no_dup.sum()

            if num_new_files == 0:
                msg = 'All {} master cals on {} already in database'
                logger.warning(msg.format(num_existing_files, date))
                return None

            new_rows = pd.DataFrame(dict(serialno=serialnos,
                                         frame_type=frame_types,
                                         exptime=exptimes,
                                         tilt=tilts))

            if num_new_files < num_existing_files:
                if num_new_files == 0:
                    warning_str = 'Skipped *ALL* {} files on {} '\
                                  'to avoid duplicates'
                else:
                    warning_str = 'Skipped {} files on {} to avoid duplicates'
                file_diff = num_existing_files - num_new_files
                logger.warning(warning_str.format(file_diff, date))

            if num_new_files > 0:
                new_rows.drop(new_rows[~no_dup].index, inplace=True)
                index = pd.RangeIndex(self.next_index,
                                      self.next_index + len(new_rows),
                                      name='frame_id')
                new_rows.set_index(index, inplace=True)
                new_rows['date'] = date
                self.df = self.df.append(new_rows, sort=False)
                msg = 'Added {} mastercals to DB'.format(len(new_rows))
                logger.info(msg)

        else:
            logger.warning('No new files found for {}'.format(date))

        return new_rows
