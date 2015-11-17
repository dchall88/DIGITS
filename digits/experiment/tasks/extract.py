# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import numpy as np
import pickle
from collections import OrderedDict

from digits.task import Task

# NOTE: Increment this everytime the picked object changes
PICKLE_VERSION = 1

class ExtractTask(Task):
    """
    Defines required methods for child classes
    """

    def __init__(self, dataset, **kwargs):
        """
        Arguments:
        dataset -- a DatasetJob containing the dataset for this experiment
        """
        super(ExtractTask, self).__init__(**kwargs)
        self.pickver_task_test = PICKLE_VERSION

        self.dataset = dataset        

    def __getstate__(self):
        state = super(ExtractTask, self).__getstate__()
        if 'dataset' in state:
            del state['dataset']
        return state

    def __setstate__(self, state):
        super(ExtractTask, self).__setstate__(state)
        self.dataset = None

