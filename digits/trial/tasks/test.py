# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import numpy as np
import pickle
from collections import OrderedDict

from digits.task import Task

# NOTE: Increment this everytime the picked object changes
PICKLE_VERSION = 1

class TestTask(Task):
    """
    Defines required methods for child classes
    """

    def __init__(self, dataset, model, **kwargs):
        """
        Arguments:
        dataset -- a DatasetJob containing the dataset for this experiment

        Keyword arguments:
        batch_size -- if set, override any network specific batch_size with this value
        """
        self.batch_size = kwargs.pop('batch_size', None)

        super(TestTask, self).__init__(**kwargs)
        self.pickver_task_test = PICKLE_VERSION

        self.dataset = dataset
        self.model = model
        self.num_test_examples = dataset.test_db_task().entries_count
        

    def __getstate__(self):
        state = super(TestTask, self).__getstate__()
        if 'dataset' in state:
            del state['dataset']
        if 'model' in state:
            del state['model']
        if 'labels' in state:
            del state['labels']
        return state

    def __setstate__(self, state):
        super(TestTask, self).__setstate__(state)
        self.model = None
        self.dataset = None


    def can_view_activations(self):
        """
        Returns True if this Task can visualize the activations of a model after inference
        """
        raise NotImplementedError()

    def inference(self, data, model_epoch=None):
        """
        Run inference
        """
        return None

    def read_labels(self):
        """
        Read labels from self.labels_file and store them at self.labels
        Returns True if at least one label was read
        """
        # The labels might be set already
        if hasattr(self, 'labels') and self.labels and len(self.labels) > 0:
            return True

        assert hasattr(self.dataset, 'labels_file'), 'labels_file not set'
        assert self.dataset.labels_file, 'labels_file not set'
        assert os.path.exists(self.dataset.path(self.dataset.labels_file)), 'labels_file does not exist'

        labels = []
        with open(self.dataset.path(self.dataset.labels_file)) as infile:
            for line in infile:
                label = line.strip()
                if label:
                    labels.append(label)

        assert len(labels) > 0, 'no labels in labels_file'

        self.labels = labels
        return True

