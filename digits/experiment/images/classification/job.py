# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
from digits.experiment import tasks
from digits import utils
from digits.utils import subclass, override
from ..job import ImageExperimentJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class ImageClassificationExperimentJob(ImageExperimentJob):
    """
    A Job that creates an image experiment for a classification network
    """

    def __init__(self, **kwargs):
        super(ImageClassificationExperimentJob, self).__init__(**kwargs)
        self.pickver_job_experiment_image_classification = PICKLE_VERSION
        

    @override
    def job_type(self):
        return 'Image Classification Experiment'
        


