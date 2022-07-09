import pandas as pd
import numpy as np
# import dfreduce.flags as dff

# lightFlags, darkFlags, flatFlags = dff.LightFlags.bit_to_str, dff.DarkFlags.bit_to_str, \
#                                     dff.FlatFlags.bit_to_str



def basic_info(data):
    """
    Gives you the number of each type of frame taken and the percentage 
    that do not have flags.
    
    This function takes a pandas dataframe of observations from the Dragonfly (DF)
    Telephoto Array and filters out the light frames, dark frames and flat frames.
    It also identifies the number of frames that have zero flags and returns the
    ratio of good frames to total frames.

    Parameters:
    -------------
    data : array-like (Pandas DataFrame)
       Total set of frames taken by the DF (already reduced).
       
    Returns:
    ------------
    int
        The number of light frames.
        
    int
        The number of dark frames.
        
    int
        The number of flat frames.
        
    float
        The percentage of good light frames.
        
    float
        The percentage of good dark frames.
        
    float
        The percentage of good flat frames.


    """
    
    
    is_light = data['frame_type']== 'light'
    is_dark = data['frame_type']== 'dark'
    is_flat = data['frame_type']== 'flat'

    light, dark, flat = data[is_light], data[is_dark], data[is_flat]

    gl, gd, gf = light['is_good'], dark['is_good'], flat['is_good']
    
    per_l, per_d, per_f = round(100 * gl.sum()/is_light.sum(), 1), round(100 * gd.sum()/is_dark.sum(), 1), round(100 * gf.sum()/is_flat.sum(), 1)
    
    return is_light.sum(), is_dark.sum(), is_flat.sum(), per_l, per_d, per_f




def frame_per_lens(data):
    """
    Organizes the observations by the individual cameras.
    
    This function takes the set of observations made on a given night
    and returns the basic info for each of the units that took frames.
    
    Parameters:
    ------------
    data: array-like (Pandas DataFrame)
        Total set of frames taken by the DF (already reduced).
        
    Returns:
    ------------
    int
        The number of units (cameras) that took frames on a chosen night.
        
    df: array-like (Pandas DataFrame)
        The DataFrame organized by DF unit and displaying the basic
        info for each unit.
    
    """
    
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
    """
    Gives you the number of each flag raised in a set of frames.
    
    This function counts the number of occurences that each flag
    is raised in a set of frames. Flags are distinguished bit-wise,
    with the length of the binary number equal to the number of 
    possible flags.
    
    Parameters:
    -------------
    flagged_frames: array-like (Pandas DataFrame)
         A DataFrame comprised of frames with raised flags
         of a unique 'frame_type'.
    
    flag_list: list (dtype='str')
        List of all the possible flag names for 
        a chosen 'frame_type'.
    
    Returns:
    ------------
    store: array-like (NumPy Array)
        An array of the number of each flag raised, separated
        bit-wise.
    
    
    """
    
    store = np.zeros(len(flag_list))
    stack_flags = list(flagged_frames['flags'])
    
    for i,j in enumerate(stack_flags):
        k =  np.binary_repr(j, width=len(flag_list))
        vals = np.array([int(a) for a in k])

        store += vals
        
    return store
    

    


def total_flags(don, flag_names):
    """
    Gives the number of flags raised in each category of frame type.
    
    This function takes a raw set of observations and will count the number
    of occurences for each flag for the light, dark and flat frames separately.
    
    Parameters:
    ------------
    don: array-like (Pandas DataFrame)
        Raw set of observations from DF.
    
    flag_names: ndarray
        Set of arrays which hold the names of the flags for each frame type.
        
    Retunrs:
    ------------
    count_light: array
        Number of each flag raised for the light frames, separated bit-wise.
    
    count_dark: array
        Number of each flag raised for the dark frames, separated bit-wise.
        
    count_flat: array
        Number of each flag raised for the flat frames, separated bit-wise.
    
    
    """

    flagged = don[don['flags'] != 0]

    lightF, darkF, flatF = flagged[flagged['frame_type'] == 'light'], flagged[flagged['frame_type'] == 'dark'], \
                        flagged[flagged['frame_type'] == 'flat']

    
    
    count_light, count_dark, count_flat = flag_count(lightF, flag_names[0]), flag_count(darkF, flag_names[1]), flag_count(flatF, flag_names[2])
    
    return count_light, count_dark, count_flat

