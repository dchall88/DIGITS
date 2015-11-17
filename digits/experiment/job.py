# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from digits import utils
from digits.job import Job
from . import tasks

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ExperimentJob(Job):
    """
    A Job that creates a neural network experiment
    """

    def __init__(self, dataset_id, model_id, layers, **kwargs):
        """
        Arguments:
        dataset_job -- the DatasetJob that this ExperimentJob depends on
        """
        super(ExperimentJob, self).__init__(**kwargs)
        self.pickver_job_dataset = PICKLE_VERSION

        self.dataset_id = dataset_id
        self.load_dataset()

        self.model_id = model_id
        self.load_model()
        self.layers = layers


    def __getstate__(self):
        state = super(ExperimentJob, self).__getstate__()
        if 'dataset' in state:
            del state['dataset']
        return state

    def __setstate__(self, state):
        super(ExperimentJob, self).__setstate__(state)
        self.dataset = None


    def load_dataset(self):
        from digits.webapp import scheduler
        job = scheduler.get_job(self.dataset_id)
        assert job is not None, 'Cannot find dataset'
        self.dataset = job
        for task in self.tasks:
            task.dataset = job
            
            
    def load_model(self):
        from digits.webapp import scheduler
        job = scheduler.get_job(self.model_id)
        assert job is not None, 'Cannot find model'
        self.model = job
 

    def extract_task(self):
        """Return the first ExtractTask for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.ExtractTask)][0]
        
    
    def extract_all_tasks(self):
        """Return the all ExtractTasks for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.ExtractTask)]

