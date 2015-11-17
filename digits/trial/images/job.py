# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from ..job import TrialJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ImageTrialJob(TrialJob):
    """
    A Job that creates an image trial
    """

    def __init__(self, **kwargs):
        """
        """
        super(ImageTrialJob, self).__init__(**kwargs)
        self.pickver_job_trial_image = PICKLE_VERSION

