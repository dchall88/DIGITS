# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import caffe
import time
import math
import subprocess
import sys

import numpy as np
from scipy.ndimage.interpolation import zoom
from google.protobuf import text_format
from caffe.proto import caffe_pb2

from extract import ExtractTask
import digits
from digits.config import config_value
from digits.status import Status
from digits import utils, dataset, model
from digits.utils import subclass, override, constants
from digits.dataset import ImageClassificationDatasetJob
from digits.model import ImageClassificationModelJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class CaffeExtractTask(ExtractTask):
    """
    Extracts values from different layers of a caffe model
    """

    CAFFE_LOG = 'caffe_output.log'

    @staticmethod
    def upgrade_network(cls, network):
        #TODO
        pass

    def __init__(self, caffe_model, batch_deploy_file, layers, results_file, num_examples, **kwargs):
        """
        Arguments:
        """
        super(CaffeExtractTask, self).__init__(**kwargs)
        self.pickver_task_caffe_test = PICKLE_VERSION

        self.caffe_model = caffe_model
        self.batch_deploy_file = batch_deploy_file
        self.layers = layers
        self.results_file = results_file
        self.num_examples = num_examples

    def __getstate__(self):
        state = super(CaffeExtractTask, self).__getstate__()

        # Don't pickle these things
        if 'caffe_log' in state:
            del state['caffe_log']

        return state

    def __setstate__(self, state):
        super(CaffeExtractTask, self).__setstate__(state)


    ### Task overrides

    @override
    def name(self):
        return 'Extract Features from Caffe Model'

    @override
    def before_run(self):
        if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
            self.save_prototxt_files()
        else:
            raise NotImplementedError()

        self.caffe_log = open(self.path(self.CAFFE_LOG), 'a')
        self.last_unimportant_update = None
        return True

    def save_prototxt_files(self):
        """
        Save test file to disk
        """

        # Get test file from model's directory

        # Save copy of test prototxt in experiments directory


        return True

    @override
    def offer_resources(self, resources):
        key = 'caffe_extract_features_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def task_arguments(self, resources, env):
        args = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(digits.__file__)), 'tools', 'caffe_extract.py'),
                self.batch_deploy_file,
                self.caffe_model,
                ' '.join(self.layers),
                self.results_file,
                self.num_examples,
                ]
        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        self.caffe_log.write('%s\n' % line)
        self.caffe_log.flush()

        # parse caffe header
        timestamp, level, message = self.preprocess_output_caffe(line)

        if not message:
            return True

        match = re.match(r'Memory required for data:\s+(\d+)', message)
        if match:
            pass

        # memory requirement
        match = re.match(r'Memory required for data:\s+(\d+)', message)
        if match:
            bytes_required = int(match.group(1))
            self.logger.debug('memory required: %s' % utils.sizeof_fmt(bytes_required))
            return True

        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        return True

    def preprocess_output_caffe(self, line):
        """
        Takes line of output and parses it according to caffe's output format
        Returns (timestamp, level, message) or (None, None, None)
        """
        # NOTE: This must change when the logging format changes
        # LMMDD HH:MM:SS.MICROS pid file:lineno] message
        match = re.match(r'(\w)(\d{4} \S{8}).*]\s+(\S.*)$', line)
        if match:
            level = match.group(1)
            # add the year because caffe omits it
            timestr = '%s%s' % (time.strftime('%Y'), match.group(2))
            message = match.group(3)
            if level == 'I':
                level = 'info'
            elif level == 'W':
                level = 'warning'
            elif level == 'E':
                level = 'error'
            elif level == 'F': #FAIL
                level = 'critical'
            timestamp = time.mktime(time.strptime(timestr, '%Y%m%d %H:%M:%S'))
            return (timestamp, level, message)
        else:
            #self.logger.warning('Unrecognized task output "%s"' % line)
            return (None, None, None)

    def send_iteration_update(self, it):
        """
        Sends socketio message about the current iteration
        """
        from digits.webapp import socketio

        if self.current_iteration == it:
            return

        self.current_iteration = it
        self.progress = float(it)/self.solver.max_iter

        socketio.emit('task update',
                {
                    'task': self.html_id(),
                    'update': 'progress',
                    'percentage': int(round(100*self.progress)),
                    'eta': utils.time_filters.print_time_diff(self.est_done()),
                    },
                namespace='/jobs',
                room=self.job_id,
                )


    @override
    def after_run(self):
        self.caffe_log.close()


    @override
    def after_runtime_error(self):
        if os.path.exists(self.path(self.CAFFE_LOG)):
            output = subprocess.check_output(['tail', '-n40', self.path(self.CAFFE_LOG)])
            lines = []
            for line in output.split('\n'):
                # parse caffe header
                timestamp, level, message = self.preprocess_output_caffe(line)

                if message:
                    lines.append(message)
            # return the last 20 lines
            self.traceback = '\n'.join(lines[len(lines)-20:])

