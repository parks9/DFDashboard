# Third-party 
import pandas as pd
import numpy as np

# Project
from dfreduce import DarkFlags, FlatFlags, LightFlags, FlagArray


def test_flag_objects():
    flags_1 = DarkFlags()
    flags_2 = DarkFlags(set_flags='NEGATIVE_MEDIAN,LOW_RMS')
    assert flags_1.bin_val == '0000000'
    assert flags_2.bin_val == '0001001'
    assert flags_2.to_string() == 'NEGATIVE_MEDIAN,LOW_RMS'

    flags_1 = FlatFlags(set_flags='MOON_UP')
    flags_2 = FlatFlags()
    flags_2.set('SKIPPED_TWILIGHT_COMPARISON,BAD_GROUP,NEGATIVE_PIXELS')
    assert flags_1.bin_val == '0000000000000000001'
    assert flags_2.bin_val == '0000001000001000100'
    assert flags_1.to_string() == 'MOON_UP'
    assert flags_2.to_bit_array().sum() == 3

    flags = LightFlags()
    for f in flags.str_to_bit.keys():
        assert flags.is_not_set(f)
        flags.set(f)
    assert flags.int_val == 2047
    assert flags.count() == 11
    assert flags.bin_val == '11111111111'

    a = np.array([True,  True, False,  True,  True, 
                  False,  True,  True,  True, True, True])
    assert np.alltrue(flags.to_mask(ignore_bits=[2, 5]) == a)

    a = np.array([True,  True, False,  True,  True,  
                  True,  True,  True,  True, False, True])
    assert np.alltrue(a == flags.to_mask(ignore_flags='HALOS,DOUBLE_STARS'))

    a = np.array([True,  True, False,  True, False,  
                  True,  True,  True,  True, True, True])
    assert np.alltrue(a == flags.to_mask(ignore_ints=[4, 16]))

    for f in flags.str_to_bit.keys():
        assert flags.is_set(f)
        flags.unset(f)

    assert flags.to_string() is None


def test_flag_arithmetic():
    flags_1 = DarkFlags()
    flags_2 = DarkFlags(34)
    flags_3 = DarkFlags(7)
    assert (flags_1 - flags_2).bin_val == '0000000'
    assert (flags_1 + flags_2).bin_val == flags_2.bin_val
    assert (flags_2 - flags_1).bin_val == flags_2.bin_val
    assert (flags_2 + flags_3).bin_val == '0100111'
    assert (flags_2 - flags_3).bin_val == '0100000'
    assert (flags_2 - flags_3 - flags_3).bin_val == '0100000'
    assert not flags_2 == flags_3
    assert (flags_2 - flags_2) == flags_1



def test_flag_array():
    f = dict(flags=np.array([255,   1,   2,   4,  16,   0, 40, 32]))
    df = pd.DataFrame(f)
    f_arr = FlagArray('light', flags=df.flags)
    assert f_arr.is_good().sum() == 1
    assert f_arr.to_string().loc[3] == 'DOUBLE_STARS'
    assert f_arr.get_good()[0] == 5
    assert f_arr.frac_set('NO_MASTER_DARK') == 0.25
