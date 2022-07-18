import os

project_dir = os.path.dirname(os.path.dirname(__file__))
package_data_dir = os.path.join(project_dir, 'data')
package_dir = os.path.join(project_dir, 'dfreduce')

data_path_default = os.getenv('RAW_DATA_PATH')
db_path_default = os.getenv('DB_PATH')
mcals_path_default = os.getenv('MCALS_PATH')
apass_path = os.getenv('APASS_PATH')

from .dfstruct import DFStruct
from . import log
from .log import logger

if data_path_default is None:
    logger.warning('Default raw data path environment variable not defined')
if db_path_default is None:
    logger.warning('Default database path environment variable not defined')
if mcals_path_default is None:
    logger.warning('Default MasterCals path environment variable not defined')
if apass_path is None:
    logger.warning('Default APASS path environment variable not defined')

from . import utils
from . import viz 
from . import modeling
from . import improc
from . import detection
from . import astrometry
from . import tap
from .viz import show_image
from .detection import sextractor
from .improc import swarp
from .astrometry import scamp
from .cameras import camera_info, filter_dict, get_filter_name

from . import tasks
from . import pipelines
from .driver import Driver
from .database import *
from .butler import *
from .flags import *
from . import cli
