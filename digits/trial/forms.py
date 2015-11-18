# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from flask.ext.wtf import Form
from wtforms import validators
from digits import utils


class TrialForm(Form):

    def __init__(self, csrf_enabled=False, *args, **kwargs):
        super(TrialForm, self).__init__(csrf_enabled=csrf_enabled, *args, **kwargs)

        
    ### Learning Method
    learning_method = utils.forms.SelectField('Select Learning Method', choices=[])
    
    ### Layers
    layers = utils.forms.SelectField('Layers', choices=[], validators = [validators.Optional()])
    
    ### Labels
    category_names = utils.forms.SelectField('Select Category Name', choices=[])
              
    ### Name      
    trial_name = utils.forms.StringField('Trial Name',
            validators = [
                validators.DataRequired()
                ]
            )

    ### SVM Parameters
    normalise = utils.forms.SelectField(u'Normalise Features',
            default='1',
            choices=[
                ('0', 'No'),
                ('1', 'Yes'),
                ]
            )
    C = utils.forms.FloatField('Penalty Parameter',
            validators = [
                validators.NumberRange(min=0)
                ],
            default=0.1,
            )
    class_weight = utils.forms.SelectField(u'Class Weight',
            default='auto',
            choices=[
                ('none', 'none'),
                ('auto', 'auto'),
                ]
            )
    
    ### AdaBoost Parameters
    adaboost_parameters = utils.forms.StringField('AdaBoost Parameters (IN DEVELOPMENT)',)
