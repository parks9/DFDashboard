import abc
import numpy as np
import pandas as pd
from .log import logger
from . import utils


__all__ = ['DarkFlags', 'FlatFlags', 'LightFlags', 'FlagArray']


def int_to_bit_array(int_val, num_bits, dtype=int):
    bits = 2**np.arange(num_bits)
    return (int_val & bits != 0).astype(dtype)


def bit_array_to_int(bit_array):
    bits = 2**np.arange(len(bit_array))
    return bits[bit_array.astype(bool)].sum()


class Flags(metaclass=abc.ABCMeta):

    frame_type = None
    str_to_bit = dict()
    bit_to_str = dict()
    str_to_int = dict()
    int_to_str = dict()

    def __init__(self, int_val=0, set_flags=None):
        self._int_val = int(int_val)
        self.num_flags = len(self.str_to_bit)
        if set_flags is not None:
            set_flags = utils.list_of_strings(set_flags)    
            self.set(set_flags)

    def _check_bit_or_str(self, bit_or_str):
        if type(bit_or_str) == str:
            bit = self.str_to_bit[bit_or_str.upper()]
        else:
            bit =  bit_or_str
        return bit

    @property
    def int_val(self):
        return int(self._int_val)

    @property
    def bin_val(self):
        return np.binary_repr(self.int_val, width=self.num_flags)

    def is_set(self, bit_or_str):
        bit = self._check_bit_or_str(bit_or_str)
        return (self.int_val & 2**bit) != 0 

    def is_not_set(self, bit_or_str):
        bit = self._check_bit_or_str(bit_or_str)
        return (self.int_val & 2**bit) == 0 

    def set(self, bit_or_str):
        if not utils.is_list_like(bit_or_str):
            if type(bit_or_str) == str:
                bit_or_str = utils.list_of_strings(bit_or_str)
            else:
                bit_or_str = [bit_or_str]
        for bs in bit_or_str:
            bit = self._check_bit_or_str(bs)
            if not self.is_set(bit):
                self._int_val += 2**bit

    def unset(self, bit_or_str):
        bit = self._check_bit_or_str(bit_or_str)
        if self.is_set(bit):
            self._int_val -= 2**bit

    def mask_to_bits(self, **kwargs):
        mask = self.to_mask(**kwargs)
        bits = np.argwhere(mask).flatten()
        return bits

    def to_mask(self, ignore_flags=[], ignore_bits=[], ignore_ints=[]):
        unmask_val = 0
        bit_mask = int_to_bit_array(self.int_val, self.num_flags, bool)
        iflags = utils.list_of_strings(ignore_flags)
        for f in iflags:
            unmask_val += 2**self._check_bit_or_str(f)
        ibits = utils.make_list_like_if_needed(ignore_bits) 
        for b in ibits:
            unmask_val += 2**int(b)
        iints = utils.make_list_like_if_needed(ignore_ints)
        for i in iints:
            unmask_val += i
        if unmask_val > 0:
            ignore_mask = int_to_bit_array(unmask_val, self.num_flags, bool)
            bit_mask[ignore_mask] = False
        return bit_mask

    def to_bit_array(self, **kwargs):
        return self.to_mask(**kwargs).astype(int)

    def to_integer(self, **kwargs):
        mask = self.to_mask(**kwargs)
        int_val = bit_array_to_int(mask)
        return int_val

    def to_binary(self, **kwargs):
        int_val = self.to_integer(**kwargs)
        bin_val = np.binary_repr(int_val, width=self.num_flags)
        return bin_val

    def to_string(self, joiner=',', lower=False, **kwargs):
        bits = self.mask_to_bits(**kwargs)
        if lower:
            flag_str = [self.bit_to_str[i].lower() for i in bits]
        else:
            flag_str = [self.bit_to_str[i] for i in bits]
        flag_str = joiner.join(flag_str) if len(flag_str) > 0 else None
        return flag_str

    def on_bits(self, **kwargs):
        return np.argwhere(self.to_bit_array(**kwargs)).flatten()

    def count(self, **kwargs):
        return self.to_mask(**kwargs).sum()

    def __add__(self, other):
        assert self.__class__ == other.__class__, 'Flag type must to match!'
        return self.__class__(self.int_val | other.int_val)

    def __sub__(self, other):
        assert self.__class__ == other.__class__, 'Flag type must to match!'
        return self.__class__(self.int_val & ~other.int_val)

    def __eq__(self, other):
        return self.int_val == other.int_val

    def __len__(self):
        return self.to_mask().sum()

    def __repr__(self):
        repr = '< {} Flags | {} >'
        return repr.format(self.frame_type.upper(), self.to_binary())


class DarkFlags(Flags):

    frame_type = 'dark'
    str_to_bit = dict(
        NEGATIVE_MEDIAN=0,
        HIGH_MEDIAN=1,
        LOW_MEDIAN=2,
        LOW_RMS=3,
        HIGH_RMS=4,
        HIGH_DARK_OVER_MODEL=5,
        HIGH_MODEL_VARIATION=6,
        ZERO_PIXEL_FRACTION=7,
        LARGE_VALUE_RANGE=8,
        CORRUPT_FILE=9
    )
    bit_to_str = dict(map(reversed, str_to_bit.items()))
    str_to_int = {k: int(2**v) for k, v in str_to_bit.items()}
    int_to_str = dict(map(reversed, str_to_int.items()))


class FlatFlags(Flags):

    frame_type = 'flat'
    str_to_bit = dict(
        MOON_UP=0,
        MOON_NEAR=1,
        NEGATIVE_PIXELS=2,
        LOW_MEDIAN_COUNTS=3,
        HIGH_MEDIAN_COUNTS=4,
        BAD_SLOPE=5,
        BAD_GROUP=6,
        HIGH_GROUP_RAMP_STDDEV=7,
        HIGH_GOOD_GROUP_RAMP_STDDEV=8,
        NO_GOOD_RAMPS=9,
        BAD_TWILIGHT_MASTER_SLOPE=10,
        BAD_BY_ASSOC=11,
        SKIPPED_TWILIGHT_COMPARISON=12,
        BAD_HEADER_ALTAZ=13,
        PART_OF_DAY_CONFUSION=14,
        CHANGED_PART_OF_DAY=15,
        MOON_CONFUSION=16,
        CHANGED_MOON_STATUS=17,
        NO_MASTER_DARK=18, 
        ZERO_PIXEL_FRACTION=19,
        CORRUPT_FILE=20
    )

    bit_to_str = utils.reverse_dict(str_to_bit)
    str_to_int = {k: int(2**v) for k, v in str_to_bit.items()}
    int_to_str = utils.reverse_dict(str_to_int)
    bookkeeping_strs = [
        'SKIPPED_TWILIGHT_COMPARISON',
        'BAD_HEADER_ALTAZ',
        'PART_OF_DAY_CONFUSION',
        'CHANGED_PART_OF_DAY',
        'MOON_CONFUSION',
        'CHANGED_MOON_STATUS'
    ]

    @property
    def bookkeeping_bits(self):
        return [self.str_to_bit[s] for s in self.bookkeeping_strs]

    @property
    def bookkeeping_ints(self):
        return [int(2**b) for b in self.bookkeeping_bits]
        

class LightFlags(Flags):

    frame_type = 'light'
    str_to_bit = dict(
        NO_MASTER_DARK=0,
        NO_MASTER_FLAT=1,
        DOUBLE_STARS=2,
        BAD_FOCUS=3,
        TOO_FEW_OBJECTS=4,
        HIGH_ELLIPTICITY=5,
        OFF_TARGET=6,
        OFF_HEADER_TARGET=7,
        ASTROMETRY_FAILED=8,
        HALOS=9,
        TOO_FEW_REF_SOURCES=10,
        SOURCE_ASYMMETRY=11, 
        CORRUPT_FILE=12
    )
    bit_to_str = dict(map(reversed, str_to_bit.items())) 
    str_to_int = {k: int(2**v) for k, v in str_to_bit.items()}
    int_to_str = dict(map(reversed, str_to_int.items()))


class FlagArray(object):

    flag_dict = dict(
        dark=DarkFlags,
        flat=FlatFlags,
        light=LightFlags
    )

    def __init__(self, frame_type, values=None, index=None, flags=None):
        types = list(self.flag_dict.keys())
        msg = f"'{frame_type}' is not a valid frame type! "
        self.frame_type = frame_type
        assert frame_type in types, msg + 'Valid types = ' + ', '.join(types)
        if flags is not None:
            if type(flags) == pd.Series:
                index = flags.index
                values = flags.values
            elif type(flags) == pd.DataFrame:
                index = flags.index
                values = flags.values.flatten().astype(int)
            else:
                raise Exception('flags must be pd.Series or pd.DataFrame')
        elif index is not None:
            assert type(index) == pd.Int64Index, 'index must be pd.Int64Index'
            values = values if values is not None else [0]
        else:
            values = values if values is not None else [0]

        self.df = pd.DataFrame(dict(flags=values), index=index)
        self.Flags = self.flag_dict[frame_type]
        self.num_flags = len(self.Flags.str_to_bit)
        self.index = index

    def _filter_na(self, attr, name, na_val=pd.NA, **kwargs):
        results = []
        for idx, flags in self.df['flags'].iteritems():
            if pd.isna(flags):
                results.append(na_val)
            else:
                results.append(getattr(self.Flags(flags), attr)(**kwargs))
        results = pd.Series(results, index=self.index, name=name)
        return results

    @classmethod
    def from_pickle(cls, file_name):
        return utils.load_pickled_data(file_name)

    def count(self, name='num_flags', na_val=pd.NA, **kwargs):
        return self._filter_na('count', name, na_val, **kwargs)

    def is_good(self, **kwargs):
        return self.count(na_val=99, name='is_good', **kwargs) == 0

    def to_binary(self, na_val=pd.NA, **kwargs):
        return self._filter_na('to_binary', 'flags', na_val, **kwargs) 

    def to_string(self, na_val=pd.NA, **kwargs):
        return self._filter_na('to_string', 'flags', na_val, **kwargs) 

    def to_integer(self, na_val=pd.NA, **kwargs):
        return self.df

    def is_set(self, bit_or_str, na_val=pd.NA):
        return self._filter_na('is_set', 'is_set', 
                                na_val, bit_or_str=bit_or_str)

    def is_not_set(self, bit_or_str, na_val=pd.NA):
        return self._filter_na('is_not_set', 'is_not_set', 
                               na_val, bit_or_str=bit_or_str)

    def frac_set(self, bit_or_str):
        is_set = self._filter_na('is_set', 'is_set', 
                                 np.nan, bit_or_str=bit_or_str)
        return np.nansum(is_set) / len(self.df)

    def get_good(self, **kwargs):
        return self.loc[self.is_good(**kwargs)].index

    def to_pickle(self, file_name, **kwargs):
        utils.pickle_data(file_name, self, **kwargs)

    def __len__(self):
        return len(self.df)

    def __add__(self, other):
        assert self.__class__ == other.__class__
        assert self.frame_type == other.frame_type, 'Frame types must match!'
        flags = self.df.flags | other.df.flags
        return self.__class__(self.frame_type, self.index, flags.values)

    def __sub__(self, other):
        assert self.__class__ == other.__class__
        assert self.frame_type == other.frame_type, 'Frame types must match!'
        flags = self.df.flags & ~other.df.flags
        return self.__class__(self.frame_type, self.index, flags.values)

    def __eq__(self, other):
        return np.allclose(self.df.flags, other.df.flags)

    @property
    def loc(self):
        return self.df.loc

    @property
    def flags(self):
        return self.df.flags

    def set(self, idx, bit_or_str):
        f = self.Flags()
        f.set(bit_or_str)
        self.loc[idx, 'flags'] |= f.to_integer()

    def unset(self, idx, bit_or_str):
        f = self.Flags()
        f.set(bit_or_str)
        self.loc[idx, 'flags'] &= ~f.to_integer()

    def __repr__(self):
        repr = '<{} FlagArray | {}>'
        return repr.format(self.frame_type.upper(), self.df.flags.__repr__())
