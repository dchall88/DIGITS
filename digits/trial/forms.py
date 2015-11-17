# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os

from flask.ext.wtf import Form
import wtforms
from wtforms import validators

class TrialForm(Form):

    def __init__(self, csrf_enabled=False, *args, **kwargs):
        super(TrialForm, self).__init__(csrf_enabled=csrf_enabled, *args, **kwargs)

        
    ### Learning Method
    learning_method = wtforms.SelectField('Select Learning Method', choices=[])
    
    ### Layers
    layers = wtforms.SelectField('Layers', choices=[], validators = [validators.Optional()])
    
    ### Labels
    category_names = wtforms.SelectField('Select Category Name', choices=[])
              
    ### Name      
    trial_name = wtforms.StringField('Trial Name',
            validators = [
                validators.Required()
                ]
            )

    ### SVM Parameters
    normalise = wtforms.SelectField(u'Normalise Features',
            default='1',
            choices=[
                ('0', 'No'),
                ('1', 'Yes'),
                ]
            )
    C = wtforms.FloatField('Penalty Parameter',
            validators = [
                validators.NumberRange(min=0)
                ],
            default=0.1,
            )
    class_weight = wtforms.SelectField(u'Class Weight',
            default='auto',
            choices=[
                ('none', 'none'),
                ('auto', 'auto'),
                ]
            )
    
    ### AdaBoost Parameters
    adaboost_parameters = wtforms.StringField('AdaBoost Parameters',)
