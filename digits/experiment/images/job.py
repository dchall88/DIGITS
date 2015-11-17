# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from ..job import ExperimentJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ImageExperimentJob(ExperimentJob):
    """
    A Job that creates an image experiment
    """

    def __init__(self, **kwargs):
        """
        """
        super(ImageExperimentJob, self).__init__(**kwargs)
        self.pickver_job_experiment_image = PICKLE_VERSION

