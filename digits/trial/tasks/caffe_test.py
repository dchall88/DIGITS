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

from test import TestTask
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
class CaffeTestTask(TestTask):
    """
    Tests a caffe model
    """

    CAFFE_LOG = 'caffe_output.log'

    @staticmethod
    def upgrade_network(cls, network):
        #TODO
        pass

    def __init__(self, caffe_model, **kwargs):
        """
        Arguments:
        """
        self.use_gpu = kwargs.pop('use_gpu', False)
        self.get_layers = kwargs.pop('get_layers' , [])

        super(CaffeTestTask, self).__init__(**kwargs)
        self.pickver_task_caffe_test = PICKLE_VERSION

        self.caffe_model = caffe_model
        self.train_val_file = constants.CAFFE_TRAIN_VAL_FILE
        self.test_file = constants.CAFFE_TEST_FILE
        self.results_file = constants.RESULTS_FILE

    def __getstate__(self):
        state = super(CaffeTestTask, self).__getstate__()

        # Don't pickle these things
        if 'caffe_log' in state:
            del state['caffe_log']

        return state

    def __setstate__(self, state):
        super(CaffeTestTask, self).__setstate__(state)


    ### Task overrides

    @override
    def name(self):
        return 'Test Caffe Model'

    @override
    def before_run(self):
        if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
            assert self.read_labels(), 'could not read labels'
            self.save_prototxt_file()
        else:
            raise NotImplementedError()

        self.caffe_log = open(self.path(self.CAFFE_LOG), 'a')
        self.last_unimportant_update = None
        return True

    def save_prototxt_file(self):
        """
        Save test file to disk
        """

        # Get test file from model's directory

        # Update batch size if necessary

        # Save copy of test prototxt in experiments directory


        return True

    @override
    def offer_resources(self, resources):
        key = 'caffe_test_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def task_arguments(self, resources):

        args = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(digits.__file__)), 'tools', 'caffe_test.py'),
                self.model.path(self.test_file),
                self.caffe_model,
                constants.RESULTS_FILE,
                self.num_test_examples,
                ]

        #if self.use_gpu:
        #    args.append('--use_gpu')

        if self.get_layers:
            args.append('--get_layers %s' % ' '.join(get_layers))

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

    def send_data_update(self, important=False):
        """
        Send socketio updates with the latest graph data

        Keyword arguments:
        important -- if False, only send this update if the last unimportant update was sent more than 5 seconds ago
        """
        from digits.webapp import socketio

        if not important:
            if self.last_unimportant_update and (time.time() - self.last_unimportant_update) < 5:
                return
            self.last_unimportant_update = time.time()

        # loss graph data
        data = self.loss_graph_data()
        if data:
            socketio.emit('task update',
                    {
                        'task': self.html_id(),
                        'update': 'loss_graph',
                        'data': data,
                        },
                    namespace='/jobs',
                    room=self.job_id,
                    )

        # lr graph data
        data = self.lr_graph_data()
        if data:
            socketio.emit('task update',
                    {
                        'task': self.html_id(),
                        'update': 'lr_graph',
                        'data': data,
                        },
                    namespace='/jobs',
                    room=self.job_id,
                    )

    def send_snapshot_update(self):
        """
        Sends socketio message about the snapshot list
        """
        from digits.webapp import socketio

        socketio.emit('task update',
                {
                    'task': self.html_id(),
                    'update': 'snapshots',
                    'data': self.snapshot_list(),
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

    ### TestTask overrides

    def get_layer_visualization(self, layer,
            max_width=500,
            ):
        """
        Returns (activations, params) for the given layer:
            activations -- a vis_square for the activation blobs
            weights -- a vis_square for the learned weights (may be None for some layer types)
        Returns (None, None) if an error occurs

        Note: This should only be called directly after the classifier has classified an image (so the blobs are valid)

        Arguments:
        layer -- the layer to visualize

        Keyword arguments:
        max_width -- the maximum width for vis_squares
        """
        if not self.loaded_model():
            return None, None

        activations = None
        weights = None

        normalize = True
        # don't normalize softmax layers
        if layer.type == 'Softmax':
            normalize = False

        if (not layer.bottom or layer.bottom[0] != layer.top[0]) and layer.top[0] in self.classifier.blobs:
            blob = self.classifier.blobs[layer.top[0]]
            assert blob.data.ndim == 4, 'expect blob.data.ndim == 4'
            if blob.data.shape[0] == 10:
                # 4 is the center crop (if oversampled)
                data = blob.data[4]
            else:
                data = blob.data[0]

            if data.shape[0] == 3:
                # can display as color channels
                # (1,height,width,channels)
                data = data.transpose(1,2,0)
                data = data[np.newaxis,...]

            # chop off data so that it will fit within max_width
            width = data.shape[2]
            if width > max_width:
                data = data[np.newaxis,0,:max_width,:max_width]
            else:
                if width > 1:
                    padsize = 1
                    width += 1
                else:
                    padsize = 0
                n = max_width/width
                n *= n
                data = data[:n]

            activations = utils.image.vis_square(data,
                    padsize     = padsize,
                    normalize   = normalize,
                    )
        if layer.name in self.classifier.params:
            params = self.classifier.params[layer.name][0]
            assert params.data.ndim == 4, 'expect params.ndim == 4'
            data = params.data
            if data.shape[1] == 3:
                # can display as color channels
                data = data.transpose(0,2,3,1)
            else:
                data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3]))

            # chop off data so that it will fit within max_width
            width = data.shape[2]
            if width >= max_width:
                data = data[np.newaxis,0,:max_width,:max_width]
            else:
                if width > 1:
                    padsize = 1
                    width += 1
                else:
                    padsize = 0
                n = max_width/width
                n *= n
                data = data[:n]

            weights = utils.image.vis_square(data,
                    padsize     = padsize,
                    normalize   = normalize,
                    )
        return activations, weights


