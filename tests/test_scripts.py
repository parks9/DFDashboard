# Standard libray
import os, shutil
from glob import glob
from subprocess import call

# Third-party
import pytest
import numpy as np
import pandas as pd

# Project
import dfreduce
from dfreduce import DFDatabaseHub


db_path = '/tmp'
data_path = '/media/dragonfly/NAS/TestData'
mcals_path = '/media/dragonfly/MyBook/TestMasterCals'
config_fn = '/home/dragonfly/codebase/configs/testing.yml'
date = '2019-11-24'
df_unit = 101
nproc = 4


def test_reduce_night():
    cmd = f'reduce-night -c {config_fn} --date {date} --nproc {nproc}'
    call(cmd, shell=True)
    master_dark_files = glob(f'{mcals_path}/master_darks/{date}/*')
    master_flat_files = glob(f'{mcals_path}/master_flats/{date}/*')
    assert len(master_dark_files) == 16
    assert len(master_flat_files) == 1


@pytest.mark.dependency(depends=['test_reduce_night'])
def test_database_vs_files():
    db_hub = DFDatabaseHub(db_path, data_path, mcals_path)

    checked = db_hub.frames.query('flags', frame_type='light').flags >= 0
    assert np.alltrue(checked)

    path = os.path.join(db_hub.mcals.flat_path, date)
    mflat_files = glob(os.path.join(path, '*.fits'))
    path = os.path.join(db_hub.mcals.dark_path, date)
    mdark_files = glob(os.path.join(path, '*.fits'))

    num_mflats = len(mflat_files)
    num_mdarks = len(mdark_files)

    mflats = db_hub.mcals.query('serialno', date=date, frame_type='flat')
    mdarks = db_hub.mcals.query('serialno', date=date, frame_type='dark')

    assert len(mflats) == num_mflats
    assert len(mdarks) == num_mdarks
    where = f"frame_type == 'light' & date == '{date}' & master_flat_id > 0"
    r = db_hub.frames.query('serialno,date,master_flat_id', where)

    num_unique_serialno = len(r.serialno.unique())
    assert num_unique_serialno == num_mflats

    os.remove(db_path + '/df_database_individual_frames_uw.csv')
    os.remove(db_path + '/df_database_master_cals_uw.csv')

    for p in ['flats', 'darks']:
        path = os.path.join(mcals_path, 'master_' + p)
        shutil.rmtree(path)
