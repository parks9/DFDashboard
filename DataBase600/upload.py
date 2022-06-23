import pandas as pd

dtype = dict(
    serialno=str, df_unit=int, date=str, frame_type=str, target=str, 
    expnum=int, filter_name=str, ra=str, dec=str, alt=str, az=str, 
    ccd_temp=str, hdr_datetime=str, master_dark_date=pd.StringDtype(),
    master_dark_id=pd.Int64Dtype(), master_flat_date=pd.StringDtype(), 
    master_flat_id=pd.Int64Dtype(), flags=pd.Int64Dtype(), 
    is_good='boolean'
)

#data = pd.read_csv('~/School/SURP/DFDash/DataBase600/df_database_individual_frames_uw.csv', index_col='frame_id', dtype=dtype)

data = pd.read_csv('./DataBase600/df_database_individual_frames_uw.csv', index_col='frame_id', dtype=dtype)
