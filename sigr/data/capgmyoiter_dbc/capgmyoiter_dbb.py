from __future__ import division
from functools import partial
from itertools import product
from logbook import Logger
from . import Dataset as Base
from .. import get_data
from ... import constant


logger = Logger(__name__)


class Dataset(Base):

    name = 'capgmyoiter-dbb'
    subjects = list(range(2, 21, 2))
    gestures = list(range(1, 9))
    num_session = 2
    sessions = [1, 2]
    
    root = ('.cache/dbb')

    @classmethod
    def parse(cls, text):
        if text == 'capgmyoiter-dbb':
           return cls(root = '.cache/dbb')    

