# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import wtforms
from wtforms import validators

from ..forms import TrialForm

class ImageTrialForm(TrialForm):
    """
    Defines the form used to create a new ImageTrialJob
    """

    crop_size = wtforms.IntegerField('Crop Size',
            validators = [validators.Optional()]
            )
    use_mean = wtforms.BooleanField('Subtract Mean File',
            default = True
            )

