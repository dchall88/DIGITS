# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os.path
import re

import digits
from digits.status import Status
from digits import utils
from digits.utils import subclass, override
from digits.task import Task

# NOTE: Increment this everytime the pickled object
PICKLE_VERSION = 1

@subclass
class ParseJsonTask(Task):
    """Parses a folder into textfiles"""
    def __init__(self, json_annotation_file, train_test_split_file, **kwargs):
        """
        Arguments:
        json_annotation_file -- the JSON or PICKLE annotation file to parse
        train_test_split_file -- a file containing the videos to use for each set

        Keyword Arguments:
        get_bboxes -- whether to parse using bounding boxes in the images
        min_per_category -- minimum number of images per category
        max_per_category -- maximum number of images per category
        """
        # Take keyword arguments out of kwargs
        self.get_bboxes = kwargs.pop('get_bboxes', False)
        self.min_per_category = kwargs.pop('min_per_category', 2)
        self.max_per_category = kwargs.pop('max_per_category', None)

        super(ParseJsonTask, self).__init__(**kwargs)
        self.pickver_task_parsefolder = PICKLE_VERSION

        self.json_file = json_annotation_file
        self.train_test_split_file = train_test_split_file

        self.train_file = utils.constants.TRAIN_FILE
        self.val_file = utils.constants.VAL_FILE
        self.labels_file = utils.constants.LABELS_FILE

        ### Results

        self.train_count = None
        self.val_count = None
        self.test_count = None
        self.label_count = None


    def __getstate__(self):
        state = super(ParseJsonTask, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(ParseJsonTask, self).__setstate__(state)

    @override
    def name(self):
        sets = []
        sets.append('train')
        sets.append('val')
        sets.append('test')

        return 'Parse Json (%s)' % ('/'.join(sets))

    @override
    def html_id(self):
        sets = []
        sets.append('train')
        sets.append('val')
        sets.append('test')

        return 'task-parse-json-%s' % ('-'.join(sets))

    @override
    def offer_resources(self, resources):
        key = 'parse_json_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def task_arguments(self, resources):
        args = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(digits.__file__)), 'tools', 'parse_json.py'),
                self.json_file,
                self.train_test_split_file,
                self.labels_file,
                '--get_bboxes=%d' % self.get_bboxes,
                '--train_file=%s' % self.path(utils.constants.TRAIN_FILE),
                '--val_file=%s' % self.path(utils.constants.VAL_FILE),
                '--test_file=%s' % self.path(utils.constants.TEST_FILE),
                '--min=%s' % self.min_per_category,
                ]

        if self.max_per_category is not None:
            args.append('--max=%s' % self.max_per_category)

        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # progress
        match = re.match(r'Progress: ([-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)', message)
        if match:
            self.progress = float(match.group(1))
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
            return True

        # totals
        ##match = re.match(r'Found (\d+) images in (\d+) categories', message)
        ##if match:
        ##    self.label_count = int(match.group(2))
        ##    return True

        # splits
        match = re.match(r'Selected (\d+) for (\w+)', message)
        if match:
            if match.group(2).startswith('training'):
                self.train_count = int(match.group(1))
            elif match.group(2).startswith('validation'):
                self.val_count = int(match.group(1))
            elif match.group(2).startswith('test'):
                self.test_count = int(match.group(1))
            return True

        if level == 'warning':
            self.logger.warning('%s: %s' % (self.name(), message))
            return True
        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        return True


    @override
    def after_run(self):
        if self.job is not None:
            self.job.load_labels()
