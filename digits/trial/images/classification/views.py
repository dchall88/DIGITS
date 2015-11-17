# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import sys
import shutil
import tempfile
import random
import copy
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np
import cStringIO as StringIO
import lmdb
import caffe

from flask import render_template, request, redirect, url_for, flash, send_file, jsonify

import digits
from digits.config import config_value
from digits import utils
from digits.webapp import app, scheduler
from digits.dataset import ImageClassificationDatasetJob
from digits.model import ImageClassificationModelJob
from digits.trial import tasks
from forms import ImageClassificationTrialForm
from job import ImageClassificationTrialJob
from digits.status import Status

NAMESPACE = '/trials/images/classification'

@app.route(NAMESPACE + '/<job_id>/results/<class_example>')
def image_classification_trial_show_examples(job_id, class_example):
    job = scheduler.get_job(job_id)
        
    hit_results = job.hit_miss_results[str(class_example)]['hit']
    miss_results = job.hit_miss_results[str(class_example)]['miss']
        
    return render_template("trials/images/classification/images.html", class_example=class_example, miss_results=miss_results, hit_results=hit_results, job_id=job_id, job=job)
    

@app.route(NAMESPACE + '/display_image/<job_id>/<key>')
def display_images(job_id, key):
    # Display the Images
    job = scheduler.get_job(job_id)
    test_db_dir = job.path(job.test_db_task().db_name)
    env = lmdb.open(test_db_dir, readonly=True)
    #env = lmdb.open('/home/david/.digits/jobs/' + job_id + '/test_db', readonly=True)
 
    with env.begin() as txn:
        raw_datum = txn.get(str(key))
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)
    stream = StringIO.StringIO(datum.data)
        
    return send_file(stream, mimetype='image/png')
    
    
@app.route(NAMESPACE + '/<experiment_id>/new', methods=['GET'])
def image_classification_trial_new(experiment_id):
    
    experiment = load_experiment(experiment_id)
    experiment.load_dataset()
    experiment.load_model()
    
    form = ImageClassificationTrialForm()
    form.learning_method.choices = get_learning_methods()
    form.learning_method.default = get_default_learning_method()
    form.layers.choices = get_layers(copy.copy(experiment.layers))
    form.category_names.choices = get_category_names(experiment)
    return render_template('trials/images/classification/new.html', form=form, experiment=experiment)

@app.route(NAMESPACE + '/<experiment_id>/', methods=['POST'])
def image_classification_trial_create(experiment_id):

    experiment = load_experiment(experiment_id)
    experiment.load_dataset()
    experiment.load_model()

    form = ImageClassificationTrialForm()
    form.learning_method.choices = get_learning_methods()
    form.learning_method.default = get_default_learning_method()
    form.layers.choices = get_layers(copy.copy(experiment.layers))
    form.category_names.choices = get_category_names(experiment)
    
    if not form.validate_on_submit():
        return render_template('trials/images/classification/new.html', form=form, experiment=experiment), 400
    
    category_index = -1
    for index, choice in enumerate(form.category_names.choices):
        if choice[0] == form.category_names.data:
            category_index = index
            break
            
    job = None
    try:
        job = ImageClassificationTrialJob(
                name        = form.trial_name.data,
                experiment_id  = experiment.id(),
                learning_method = form.learning_method.data,
                category_name = form.category_names.data,
                category_index = category_index,
                layer = form.layers.data,
                )
                
        if form.learning_method.data == 'dnn':
            dnn_trial(job, form)

        elif form.learning_method.data == 'svm':
            svm_trial(job, form)

        elif form.learning_method.data == 'adaboost':
            boosting_trial(job, form)

        scheduler.add_job(job)
        return redirect(url_for('trials_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise
        
        
def dnn_trial(job, form):
    class_labels = job.experiment.dataset.labels[form.category_names.data]
    ypred_file = job.experiment.path('ftrs_test/prob.jbl')
    ytrue_file = job.experiment.path('ftrs_test/label.jbl')
    job.generate_data(ypred_file, ytrue_file, class_labels)
    job.generate_hit_miss_results(class_labels)
        

def svm_trial(job, form):
       
       layer = form.layers.data.replace('/', '_')
       job.tasks.append(
            tasks.SVMTask(job_dir=job.dir(),
                                   YTrain_file=job.path('train_label.jbl'),
                                   XTrain_file=job.experiment.path('ftrs_train/' + layer + '.jbl'),
                                   XTest_file=job.experiment.path('ftrs_test/' + layer + '.jbl'),
                                   class_labels = job.experiment.dataset.labels[form.category_names.data],
                                   job = job,
                                   normalise = form.normalise.data,
                                   C = form.C.data,
                                   class_weight = form.class_weight.data,
                                  )
        )
    
def boosting_trial(job, form):
    pass
       
    
def show(job):
    """
    Called from digits.views.show_job()
    """
    return render_template('trials/images/classification/show.html', job=job)


def load_experiment(experiment_id):
    job = scheduler.get_job(experiment_id)
    assert job is not None, 'Cannot find model'
    return job
        

def get_learning_methods():
    return [
            ('dnn', 'Deep Neural Net'),
            ('svm', 'SVM'),
            ('adaboost', 'AdaBoost'),
            ]


def get_default_learning_method():
    return 'dnn'

def get_layers(layers):
    layers.remove('label')
    return [(layer, layer) for layer in layers]

def get_category_names(experiment):
    if hasattr(experiment.dataset, 'labels'):
        category_names = experiment.dataset.labels.keys()
        return [(category_name, category_name) for category_name in category_names]
    else:
        return [(None, 'None')]

