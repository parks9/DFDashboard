import pandas as pd
import numpy as np
import dfreduce.flags as dff

lightFlags, darkFlags, flatFlags = dff.LightFlags.bit_to_str, dff.DarkFlags.bit_to_str, \
                                    dff.FlatFlags.bit_to_str



def basic_info(data):
    
    is_light = data['frame_type']== 'light'
    is_dark = data['frame_type']== 'dark'
    is_flat = data['frame_type']== 'flat'

    light, dark, flat = data[is_light], data[is_dark], data[is_flat]

    gl, gd, gf = light['is_good'], dark['is_good'], flat['is_good']
    
    per_l, per_d, per_f = round(100 * gl.sum()/is_light.sum(), 1), round(100 * gd.sum()/is_dark.sum(), 1), round(100 * gf.sum()/is_flat.sum(), 1)
    
    return is_light.sum(), is_dark.sum(), is_flat.sum(), per_l, per_d, per_f




def frame_per_lens(data):
    
    group_lens = data.groupby('df_unit')

    units_listed = np.array(list(group_lens.groups.keys()))
    
    fpl = {"Light frames": [], "Dark frames": [], "Flat frames": [], "Light frame quality %": [], "Dark frame quality %":[], "Flat frame quality %": []}
    
    
    
    
    for i in units_listed:
        data_in_lens_bool = data['df_unit'] == i
        
        data_in_lens = data[data_in_lens_bool]
        
        a,b,c,d,e,f = basic_info(data_in_lens)
        
        fpl["Light frames"].append(a)
        fpl["Dark frames"].append(b)
        fpl["Flat frames"].append(c)
        fpl["Light frame quality %"].append(d)
        fpl["Dark frame quality %"].append(e)
        fpl["Flat frame quality %"].append(f)
        
        
    df = pd.DataFrame(fpl, index=units_listed)
    #index_lens = np.where(dates_listed == str(night))
    
    return len(units_listed), df





def flag_count(flagged_frames, flag_list):

    store = np.zeros(len(flag_list))
    stack_flags = list(flagged_frames['flags'])
    
    for i,j in enumerate(stack_flags):
        k =  np.binary_repr(j, width=len(flag_list))
        vals = np.array([int(a) for a in k])

        store += vals
        
    return store
    

    


def total_flags(don, flag_names):
# If a frame has flags, store the flag int in a list which we will later convert to binary or string and count 
# the number of occurences for each flag

    flagged = don[don['flags'] != 0]

    lightF, darkF, flatF = flagged[flagged['frame_type'] == 'light'], flagged[flagged['frame_type'] == 'dark'], \
                        flagged[flagged['frame_type'] == 'flat']

    
    
    count_light, count_dark, count_flat = flag_count(lightF, flag_names[0]), flag_count(darkF, flag_names[1]), flag_count(flatF, flag_names[2])
    
    return count_light, count_dark, count_flat

