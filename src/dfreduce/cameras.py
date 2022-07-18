"""
A place to store the camera names, serial numbers, and detector properties.
"""
from astropy import units as u
from . import DFStruct
from . import utils
from . import logger


__all__ = ['serialno_dict', 'name_dict', 'filter_dict', 'serialno_dict_nb',
           'uw_cameras', 'name_dict_nb', 'nb_cameras', 'camera_info', 
           'all_filter_names', 'get_filter_name']


serialno_dict = dict(
    Dragonfly101='T13080513',
    Dragonfly102='T13100592',
    Dragonfly103='T13090562',
    Dragonfly104='T13100588',
    Dragonfly105='T13100585',
    Dragonfly106='T13030365',
    Dragonfly107='T13110627',
    Dragonfly108='T13110605',
    Dragonfly109='T13110598',
    Dragonfly110='T13060460',
    Dragonfly111='T13100595',
    Dragonfly112='T13110623',
    Dragonfly113='T13090528',
    Dragonfly114='T13090526',
    Dragonfly115='T13100591',
    Dragonfly116='T13100580',
    Dragonfly117='T13110625',
    Dragonfly118='T13090552',
    Dragonfly119='T13090553',
    Dragonfly120='T13110630',
    Dragonfly121='T13110597',
    Dragonfly122='T13090570',
    Dragonfly123='T13110629',
    Dragonfly124='T13070498',
    Dragonfly201='83F010612',
    Dragonfly202='83F010820',
    Dragonfly203='83F010692',
    Dragonfly204='83F010730',
    Dragonfly205='83F010783',
    Dragonfly206='83F010784',
    Dragonfly207='83F011129',
    Dragonfly208='83F010826',
    Dragonfly209='83F010827',
    Dragonfly210='T13100590',
    Dragonfly211='T13100593',
    Dragonfly212='T13070473',
    Dragonfly213='T13090554',
    Dragonfly214='T13090564',
    Dragonfly215='T13090565',
    Dragonfly216='T13090568',
    Dragonfly217='T13090571',
    Dragonfly218='T13100579',
    Dragonfly219='T13100584',
    Dragonfly220='T13100587',
    Dragonfly221='T13110600',
    Dragonfly222='T13110621',
    Dragonfly223='T13110624',
    Dragonfly224='T13110628',
)

name_dict = dict(map(reversed, serialno_dict.items()))

filter_dict = {
    'T13080513': 'G',
    'T13100592': 'G',
    'T13090562': 'G',
    'T13100588': 'G',
    'T13100585': 'G',
    'T13030365': 'G',
    'T13110627': 'G',
    'T13110605': 'G',
    'T13110598': 'G',
    'T13060460': 'G',
    'T13100595': 'G',
    'T13110623': 'G',
    'T13090528': 'R',
    'T13090526': 'R',
    'T13100591': 'R',
    'T13100580': 'R',
    'T13110625': 'R',
    'T13090552': 'R',
    'T13090553': 'R',
    'T13110630': 'R',
    'T13110597': 'R',
    '83F011129': 'R',
    'T13090570': 'R',
    'T13110629': 'R',
    'T13070498': 'R',
    '83F010612': 'G',
    '83F010687': 'G',
    '83F010692': 'G',
    '83F010730': 'G',
    '83F010783': 'G',
    '83F010784': 'G',
    '83F010820': 'G',
    '83F010826': 'G',
    '83F010827': 'G',
    'T13100590': 'G',
    'T13100593': 'G',
    'T13070473': 'G',
    'T13090554': 'R',
    'T13090564': 'R',
    'T13090565': 'R',
    'T13090568': 'R',
    'T13090571': 'R',
    'T13100579': 'R',
    'T13100584': 'R',
    'T13100587': 'R',
    'T13110600': 'R',
    'T13110621': 'R',
    'T13110624': 'R',
    'T13110628': 'R',
    'DRAGONFLY301': 'NB?',
    'DRAGONFLY302': 'NB?',
    'DRAGONFLY303': 'NB?',
}

uw_cameras = DFStruct(gain = 0.34 * u.electron / u.adu, 
                      readnoise = 8 * u.electron,
                      serialno_dict = serialno_dict, 
                      name_dict = name_dict)

serialno_dict_nb = dict(
    Dragonfly301='DRAGONFLY301',
    Dragonfly302='DRAGONFLY302',
    Dragonfly303='DRAGONFLY303',
)

name_dict_nb = dict(map(reversed, serialno_dict_nb.items()))

nb_cameras = DFStruct(gain = 0.25 * u.electron / u.adu,
                      readnoise = 4.5 * u.electron,
                      serialno_dict = serialno_dict_nb,
                      name_dict = name_dict_nb,
                      light_pixels = (12,2212,45,2795))

camera_info = dict(UW=uw_cameras, NB=nb_cameras)


all_filter_names = ['G', 'R', 'SloanG', 'SloanR', 'H-alpha', '[NII]', 'Off']
all_filter_names_lower = [f.lower() for f in all_filter_names]


def get_filter_name(path_or_header):
    """
    Get filter name from header. If the filter name is missing from the header,
    this function uses our (hopefully up to date) knowledge of the filters that
    are on each camera (serialno).

    Parameters
    ----------
    path_or_header : str of astropy Header
        Path to fits file from which to read the header or an astropy Header.

    Returns
    -------
    filt : str
        The filter name. 
    """
    header = utils.load_path_or_header(path_or_header)
    if 'FILTER' in header.keys():
        key = 'FILTER'
    elif 'FILTNAM' in header.keys():
        key = 'FILTNAM'
    else:
        key = None
        logger.warning('FILTER keyword not in header --> using filter_dict')
        filt = filter_dict[header['SERIALNO']]
    if key is not None:
        filt = header[key]
        if filt.lower() not in all_filter_names_lower:
            logger.debug(f'FILTER  = {filt} --> using filter_dict')
            filt = filter_dict[header['SERIALNO']]
        if 'sloan' in filt.lower():
            filt = filt[-1]
    return filt
