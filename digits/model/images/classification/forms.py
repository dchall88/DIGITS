# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import wtforms
from digits import utils
from wtforms import validators

from ..forms import ImageModelForm

class ImageClassificationModelForm(ImageModelForm):
    """
    Defines the form used to create a new ImageClassificationModelJob
    """
    category_names = utils.forms.SelectField('Select Category Name',
                                         default=-1,
                                         choices=[],
                                         coerce=int,
                                         validators=[validators.InputRequired()],
                                         tooltip="Choose the category label you would like to use to train the model.")

