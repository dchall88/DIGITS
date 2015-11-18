# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import caffe
import time
import math
import subprocess
import sys

import numpy as np

from digits.task import Task

import digits
from digits.config import config_value
from digits.status import Status
from digits import utils, dataset, model
from digits.utils import subclass, override, constants

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class SVMTask(Task):
    """
    Train and test an SVM
    """

    @staticmethod
    def upgrade_network(cls, network):
        #TODO
        pass

    def __init__(self, 
                 YTrain_file,
                 XTrain_file,
                 XTest_file,
                 class_labels,
                 job, 
                 normalise,
                 C,
                 class_weight, **kwargs):
        """
        Arguments:
        """

        super(SVMTask, self).__init__(**kwargs)
        self.pickver_task_svm_train_test = PICKLE_VERSION
        
        self.YTrain_file = YTrain_file
        self.XTrain_file = XTrain_file
        self.XTest_file = XTest_file
        self.class_labels = class_labels
        self.job = job
        self.normalise = normalise
        self.C = C
        self.class_weight = class_weight

    def __getstate__(self):
        state = super(SVMTask, self).__getstate__()

        return state

    def __setstate__(self, state):
        super(SVMTask, self).__setstate__(state)


    ### Task overrides

    @override
    def name(self):
        return 'Train and Test SVM'

    @override
    def before_run(self):
        return True

    @override
    def offer_resources(self, resources):
        key = 'svm_test_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def task_arguments(self, resources, env):

        args = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(digits.__file__)), 'tools', 'svm_train_test.py'),
                self.YTrain_file,
                self.XTrain_file,
                self.XTest_file,
                self.job_dir,
                self.normalise,
                self.C,
                self.class_weight,
                ]

        return args

    @override
    def process_output(self, line):
        pass
        

    @override
    def after_run(self):
        ypred_file = self.job.path('prob.jbl')
        ytrue_file = self.job.path('test_label.jbl')
        
        self.job.generate_data(ypred_file, ytrue_file, self.class_labels) 
        self.job.generate_hit_miss_results(self.class_labels)


    @override
    def after_runtime_error(self):
        pass
   

