# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
from sklearn.externals import joblib
import numpy as np
import h5py
from digits import utils
from digits.job import Job
from . import tasks

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class TrialJob(Job):
    """
    A Job that creates a neural network trial
    """

    def __init__(self, experiment_id, learning_method, category_index, category_name, layer, **kwargs):
        """
        Arguments:
        dataset_job -- the DatasetJob that this TrialJob depends on
        """
        super(TrialJob, self).__init__(**kwargs)
        self.pickver_job_dataset = PICKLE_VERSION

        self.learning_method = learning_method
        self.experiment_id = experiment_id
        self.load_experiment()
        self.copy_gt_labels(category_index)
        self.category_name = category_name
        self.layer = layer

    def __getstate__(self):
        state = super(TrialJob, self).__getstate__()
        if 'experiment' in state:
            del state['experiment']
        return state

    def __setstate__(self, state):
        super(TrialJob, self).__setstate__(state)
        self.experiment = None


    def load_experiment(self):
        from digits.webapp import scheduler
        job = scheduler.get_job(self.experiment_id)
        assert job is not None, 'Cannot find experiment'
        self.experiment = job
        
    
    def copy_gt_labels(self, category_index):
        for subset in ['train', 'val', 'test']:
            labels_file = self.experiment.dataset.path(subset + '_' + str(category_index) + '.h5')
            labels = h5py.File(labels_file, 'r')
            labels = [x for x in labels['label']]   
            labels = np.array(labels)
            labels = labels.astype(int)
            joblib.dump(labels, self.dir() + '/' + subset + '_label.jbl')
        
    
    def test_task(self):
        """Return the first TestTask for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.TestTask)][0]

