# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os

from flask.ext.wtf import Form
from digits import utils
from wtforms import validators
from digits.utils.forms import validate_prob_layer_for_experiment

class ExperimentForm(Form):

    def __init__(self, csrf_enabled=False, *args, **kwargs):
        super(ExperimentForm, self).__init__(csrf_enabled=csrf_enabled, *args, **kwargs)

        
    ### Models
    model_snapshots = utils.forms.SelectField('Snapshots',
                validators = [validators.Optional()]
                )

    model_layers = utils.forms.SelectMultipleField('Select Layers', choices=[], validators=[validators.DataRequired(), validate_prob_layer_for_experiment()])

