# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
import os
import io
import tarfile
import zipfile

import flask
import werkzeug.exceptions

from digits import utils
from digits.webapp import app, scheduler, autodoc
from digits.utils.routing import request_wants_json, job_from_request
from digits.trial import ImageClassificationTrialJob
from digits.experiment import tasks
from forms import ImageClassificationExperimentForm
from job import ImageClassificationExperimentJob
from digits.status import Status

NAMESPACE = '/experiments/images/classification'

@app.route(NAMESPACE + '/<dataset_id>/<model_id>/new', methods=['GET'])
def image_classification_experiment_new(dataset_id, model_id):
    form = ImageClassificationExperimentForm()
    dataset = get_job(dataset_id)
    model = get_job(model_id)
    form.model_snapshots.choices = get_model_snapshots(model)
    form.model_layers.choices = get_model_layers(model)

    return flask.render_template('experiments/images/classification/new.html', form=form, dataset=dataset, model=model)

@app.route(NAMESPACE  + '/<dataset_id>/<model_id>/', methods=['POST'])
def image_classification_experiment_create(dataset_id, model_id):
    form = ImageClassificationExperimentForm()
    dataset = get_job(dataset_id)
    model = get_job(model_id)
    form.model_snapshots.choices = get_model_snapshots(model)
    form.model_layers.choices = get_model_layers(model)

    if not form.validate_on_submit():
        return flask.render_template('experiments/images/classification/new.html', form=form, dataset=dataset, model=model), 400

    num_test_examples = dataset.test_db_task().entries_count
    num_train_examples = dataset.train_db_task().entries_count
    num_val_examples = dataset.val_db_task().entries_count
    
    caffe_model = form.model_snapshots.data
    layers = form.model_layers.data
    layers.append('label')
            
    if caffe_model is None:
        raise Exception("The selected model %s is invalid!" % (caffe_model,))
    if not (os.path.exists(caffe_model)):
        raise Exception("Model does not exist. May be deleted by another user/process. Please restart the server to load the correct model details")

    batch_deploy_train_file = model.path(utils.constants.CAFFE_BATCH_DEPLOY_TRAIN_FILE)
    batch_deploy_val_file = model.path(utils.constants.CAFFE_BATCH_DEPLOY_VAL_FILE)
    batch_deploy_test_file = model.path(utils.constants.CAFFE_BATCH_DEPLOY_TEST_FILE)

    results_train_file = 'ftrs_train/'
    results_val_file = 'ftrs_val/'
    results_test_file = 'ftrs_test/'

    job = None
    try:
        job = ImageClassificationExperimentJob(
                name='Experiment',
                dataset_id=dataset.id(),
                model_id=model.id(),
                layers=layers,
                )

        job.tasks.append(
                tasks.CaffeExtractTask(job_dir=job.dir(),
                                       dataset=dataset,
                                       caffe_model=caffe_model,
                                       batch_deploy_file=batch_deploy_train_file,
                                       layers=layers,
                                       results_file=results_train_file,
                                       num_examples=num_train_examples,
                                      )
        )

        job.tasks.append(
                tasks.CaffeExtractTask(job_dir=job.dir(),
                                       dataset=dataset,
                                       caffe_model=caffe_model,
                                       batch_deploy_file=batch_deploy_val_file,
                                       layers=layers,
                                       results_file=results_val_file,
                                       num_examples=num_val_examples,
                                      )
        )

        job.tasks.append(
                tasks.CaffeExtractTask(job_dir=job.dir(),
                                       dataset=dataset,
                                       caffe_model=caffe_model,
                                       batch_deploy_file=batch_deploy_test_file,
                                       layers=layers,
                                       results_file=results_test_file,
                                       num_examples=num_test_examples,
                                      )
        )

        scheduler.add_job(job)
        return flask.redirect(flask.url_for('experiments_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise

@app.route(NAMESPACE + '/<experiment_id>/new', methods=['GET'])
def image_classification_experiment_add_layers(experiment_id):
    form = ImageClassificationExperimentForm()
    experiment = get_job(experiment_id)
    experiment.load_dataset()    
    experiment.load_model()
    dataset = experiment.dataset
    model = experiment.model
    form.model_snapshots.choices = get_model_snapshots(model)
    form.model_layers.choices = get_more_model_layers(model, experiment)

    return flask.render_template('experiments/images/classification/more.html', form=form, experiment=experiment, dataset=dataset, model=model)


@app.route(NAMESPACE  + '/<experiment_id>/', methods=['POST'])
def image_classification_experiment_add_layers_create(experiment_id):
    form = ImageClassificationExperimentForm()
    job = get_job(experiment_id)
    job.load_model()
    job.load_dataset()
    model = job.model
    dataset = job.dataset

    form.model_snapshots.choices = get_model_snapshots(model)
    form.model_layers.choices = get_more_model_layers(model, job)

    if not form.validate_on_submit():
        return flask.render_template('experiments/images/classification/more.html', form=form, experiment=job, dataset=dataset, model=model), 400

    num_test_examples = dataset.test_db_task().entries_count
    num_train_examples = dataset.train_db_task().entries_count
    num_val_examples = dataset.val_db_task().entries_count
    
    caffe_model = form.model_snapshots.data
    layers = form.model_layers.data
            
    if caffe_model is None:
        raise Exception("The selected model %s is invalid!" % (caffe_model,))
    if not (os.path.exists(caffe_model)):
        raise Exception("Model does not exist. May be deleted by another user/process. Please restart the server to load the correct model details")

    batch_deploy_train_file = model.path(utils.constants.CAFFE_BATCH_DEPLOY_TRAIN_FILE)
    batch_deploy_val_file = model.path(utils.constants.CAFFE_BATCH_DEPLOY_VAL_FILE)
    batch_deploy_test_file = model.path(utils.constants.CAFFE_BATCH_DEPLOY_TEST_FILE)

    results_train_file = 'ftrs_train/'
    results_val_file = 'ftrs_val/'
    results_test_file = 'ftrs_test/'

    try:
        job.layers.extend(layers)
        job.tasks.append(
                tasks.CaffeExtractTask(job_dir=job.dir(),
                                       dataset=dataset,
                                       caffe_model=caffe_model,
                                       batch_deploy_file=batch_deploy_train_file,
                                       layers=layers,
                                       results_file=results_train_file,
                                       num_examples=num_train_examples,
                                      )
        )

        job.tasks.append(
                tasks.CaffeExtractTask(job_dir=job.dir(),
                                       dataset=dataset,
                                       caffe_model=caffe_model,
                                       batch_deploy_file=batch_deploy_val_file,
                                       layers=layers,
                                       results_file=results_val_file,
                                       num_examples=num_val_examples,
                                      )
        )

        job.tasks.append(
                tasks.CaffeExtractTask(job_dir=job.dir(),
                                       dataset=dataset,
                                       caffe_model=caffe_model,
                                       batch_deploy_file=batch_deploy_test_file,
                                       layers=layers,
                                       results_file=results_test_file,
                                       num_examples=num_test_examples,
                                      )
        )

        job.status = Status.WAIT
        return flask.redirect(flask.url_for('experiments_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise


@app.route(NAMESPACE + '/<job_id>/download',
        methods=['GET', 'POST'],
        defaults={'extension': 'tar.gz'})
@app.route(NAMESPACE + '/<job_id>/download.<extension>',
        methods=['GET', 'POST'])
@autodoc('experiments')
def features_download(job_id, extension):
    """
    Return a tarball of all features extracted from the model
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    b = io.BytesIO()
    if extension in ['tar', 'tar.gz', 'tgz', 'tar.bz2']:
        # tar file
        mode = ''
        if extension in ['tar.gz', 'tgz']:
            mode = 'gz'
        elif extension in ['tar.bz2']:
            mode = 'bz2'
        with tarfile.open(fileobj=b, mode='w:%s' % mode) as tf:
            tf.add(job.dir(), arcname='features')
    elif extension in ['zip']:
        with zipfile.ZipFile(b, 'w') as zf:
            zf.write(job.dir(), arcname='features')
    else:
        raise werkzeug.exceptions.BadRequest('Invalid extension')

    response = flask.make_response(b.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=%s.%s' % (job.id(), extension)
    return response    
    
def show(job):
    """
    Called from digits.views.show_job()
    """
    running_trials = get_trial_list(job.id(), True),
    completed_trials = get_trial_list(job.id(), False),
    return flask.render_template('experiments/images/classification/show.html',
                           job=job,
                           running_trials=running_trials,
                           completed_trials=completed_trials,)

def get_trial_list(experiment_id, running):
    return [j for j in sorted(
        [j for j in scheduler.jobs if isinstance(j, ImageClassificationTrialJob) 
                                      and (j.status.is_running() == running)
                                      and (j.experiment_id == experiment_id)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]
        
        
def get_job(job_id):
    return scheduler.get_job(job_id)
    

def get_model_snapshots(model):
    model_snapshots = [(filename, 'Epoch #%s' % epoch)
                for filename, epoch in reversed(model.train_task().snapshots)]
    return model_snapshots


def get_model_layers(model):
    if hasattr(model.train_task(), 'layers'):
        model_layers = [(layer, '%s' % layer)
                for layer in model.train_task().layers]     
    else:
        model_layers = [(None, 'None')]
    return model_layers


def get_more_model_layers(model, experiment):
    if hasattr(model.train_task(), 'layers'):
        model_layers = [(layer, '%s' % layer)
                for layer in model.train_task().layers if layer not in experiment.layers]     
    else:
        model_layers = [(None, 'None')]
    return model_layers
