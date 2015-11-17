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
from forms import ExperimentForm
import images.views
import images as experiment_images

NAMESPACE = '/experiments/'

### CRUD Routes

@app.route(NAMESPACE + 'new', methods=['GET'])
def experiments_new():
    form = ExperimentForm()
    return render_template('experiments/new.html', form=form)

@app.route(NAMESPACE, methods=['POST'])
def experiments_create():
    form = ExperimentForm()
    if form.validate_on_submit():
        return 'Yay!'
    else:
        return render_template('experiments/new.html', form=form)

@app.route(NAMESPACE + '<job_id>', methods=['GET'])
def experiments_show(job_id):
    job = scheduler.get_job(job_id)

    if job is None:
        abort(404)

    if isinstance(job, experiment_images.ImageClassificationExperimentJob):
        return experiment_images.classification.views.show(job)
    else:
        abort(404)
        
