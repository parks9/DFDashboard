from copy import deepcopy
import pandas as pd
from .database import DFDatabaseHub
from . import utils
from . import logger


class Driver(object):
    """
    Class to drive a pipeline.
    """

    def __init__(self, config):
        if type(config) == str:
            config = utils.read_config(config)
        _config = deepcopy(config)
        self._config = deepcopy(config)

        logger.info('Initializing Driver')
        self.db_path = config['db_path']
        self.data_path = config['data_path']
        self.mcals_path = config['mcals_path']
        self.checkpoints_path = _config.pop('checkpoints_path', '/tmp')
        self.delete_twilight_flats = _config.pop('delete_twilight_flats', True)
        self.trunk = None
        self.stuff_in_trunk = False
        self.completed_steps = []
        self.db_hub = DFDatabaseHub(self.db_path, 
                                    self.data_path, 
                                    self.mcals_path)
    @classmethod
    def from_pickle(cls, filename):
        logger.info('Loading Driver from pickle: {filename}')
        return utils.load_pickled_data(filename)

    def get_config(self, which=None):
        config = deepcopy(self._config)
        if which is not None:
            config = config.pop(which, {})
        return config

    def is_complete(self, step):
        return step in self.completed_steps

    def create_empty_trunk(self, index):
        logger.debug('Creating empty trunk to hold stuff')
        if self.trunk is not None:
            logger.warning('Trunk already exists. Will overwrite.')
        self.trunk = pd.DataFrame(index=index)
        self.stuff_in_trunk = True

    def delete_trunk(self):
        self.trunk = None
        self.stuff_in_trunk = False

    def add_completed_step(self, step, checkpoint_fn=None):
        self.completed_steps.append(step)
        if checkpoint_fn is not None:
            self.to_pickle(checkpoint_fn, warn_overwrite=False)

    def add_attribute(self, name, value):
        setattr(self, name, value)

    def to_pickle(self, filename, **kwargs):
        utils.pickle_data(filename, self, **kwargs)
