# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import json
import math

from google.protobuf import text_format
from flask import render_template, request, redirect, url_for, flash, make_response, abort
from caffe.proto import caffe_pb2
import caffe.draw

import digits
from digits import utils
from digits.webapp import app, scheduler
from forms import TrialForm
import images.views
import images as trial_images

NAMESPACE = '/trials/'

### CRUD Routes

@app.route(NAMESPACE + 'new', methods=['GET'])
def trials_new():
    form = TrialForm()
    return render_template('trials/new.html', form=form)

@app.route(NAMESPACE, methods=['POST'])
def trials_create():
    form = TrialForm()
    if form.validate_on_submit():
        return 'Yay!'
    else:
        return render_template('trials/new.html', form=form)

@app.route(NAMESPACE + '<job_id>', methods=['GET'])
def trials_show(job_id):
    job = scheduler.get_job(job_id)

    if job is None:
        abort(404)

    if isinstance(job, trial_images.ImageClassificationTrialJob):
        return trial_images.classification.views.show(job)
    else:
        abort(404)
        
