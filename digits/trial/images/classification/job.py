# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import pickle
import numpy as np
from collections import OrderedDict

from bokeh.embed import components
from bokeh.plotting import figure, hplot
from bokeh.charts import Bar, HeatMap
from bokeh.models import HoverTool, ColumnDataSource, TapTool, OpenURL
from bokeh.models import Range1d

from sklearn import metrics
from sklearn.externals import joblib
from scipy.stats import mode

from digits.trial import tasks
from digits import utils
from digits.utils import subclass, override
from ..job import ImageTrialJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class ImageClassificationTrialJob(ImageTrialJob):
    """
    A Job that creates an image trial for a classification network
    """

    def __init__(self, **kwargs):
        super(ImageClassificationTrialJob, self).__init__(**kwargs)
        self.pickver_job_trial_image_classification = PICKLE_VERSION
        
        # graph data
        self.confusion_matrix = None
        self.scores = None
        self.track_confusion_matrix = None
        self.track_scores = None
        self.class_labels = None
        self.hit_miss_results = None

    @override
    def job_type(self):
        return 'Image Classification Trial'
        
        
    # Plotting Functions   
    def generate_hit_miss_results(self, class_labels):
        
        if self.learning_method == 'dnn':
            prob = joblib.load(self.experiment.path('ftrs_test/prob.jbl'))
            ypred = np.argmax(prob, axis=1)    
        else:
            prob = joblib.load(self.path('prob.jbl'))
            ypred = joblib.load(self.path('ypred.jbl'))
        
        ytrue = joblib.load(self.path('test_label.jbl')) 
        prob = np.max(prob, axis=1)
        
        self.hit_miss_results = {class_label: {'hit': [], 'miss': []} for class_label in class_labels}
        for class_label_index, class_label in enumerate(class_labels):
        
            bool_index = ytrue == class_label_index
            index = np.where(bool_index)[0]
            
            ytrue1 = ytrue[bool_index]
            ypred1 = ypred[bool_index]
            prob1 = prob[bool_index]
            
            hit_ind = np.logical_and(ytrue1 == ypred1, ytrue1 == class_label_index)
            hit_ind1 = np.where(hit_ind)[0]
            miss_ind = np.logical_and(ytrue1 != ypred1, ytrue1 == class_label_index)
            miss_ind1 = np.where(miss_ind)[0]
            
            prob_hit = prob1[hit_ind]
            hit = index[hit_ind]
            
            prob_hit_ind_sorted = (-prob_hit).argsort() 
            hit = hit[prob_hit_ind_sorted]
            prob_hit = prob_hit[prob_hit_ind_sorted]
            hit_ind1 = hit_ind1[prob_hit_ind_sorted]
            
            prob_miss = prob1[miss_ind]
            miss = index[miss_ind]
            ypred_miss = ypred1[miss_ind]
            
            prob_miss_ind_sorted = (-prob_miss).argsort() 
            miss = miss[prob_miss_ind_sorted]
            ypred_miss = ypred_miss[prob_miss_ind_sorted]
            ypred_miss = ypred_miss.astype(np.int)
            prob_miss = prob_miss[prob_miss_ind_sorted]
            miss_ind1 = miss_ind1[prob_miss_ind_sorted]
            
            for miss1, ypred_miss1, prob_miss1, index1 in zip(miss, ypred_miss, prob_miss, miss_ind1):
                key = '%08d' % (miss1,)
                predicted_label = class_labels[ypred_miss1]
                score = prob_miss1
                self.hit_miss_results[class_label]['miss'].append((predicted_label, score, key, index1))
                
            for hit1, prob_hit1, index1 in zip(hit, prob_hit, hit_ind1):
                key = '%08d' % (hit1,)
                score = prob_hit1
                self.hit_miss_results[class_label]['hit'].append((score, key, index1))
            
            
    def generate_data(self, y_pred_file, y_true_file, class_labels):
        
        # Generate Standard Confusion Matrix Results
        if not (os.path.exists(y_pred_file) and os.path.exists(y_true_file)):
            return False
        
        self.class_labels = class_labels

        y_pred = joblib.load(y_pred_file)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = joblib.load(y_true_file)
        y_true = y_true.astype(int)
        self.confusion_matrix, self.scores = self.generate_confusion_matrix(y_pred, y_true)

        # Generate Results incorporating temporal information
        per_track_results = self.generate_per_track_results_mode(y_pred, y_true)
        if per_track_results is not None:
            y_pred, y_true = per_track_results
            self.track_confusion_matrix, self.track_scores = self.generate_confusion_matrix(y_pred, y_true)
        
        return True

    def generate_per_track_results_mode(self, y_pred, y_true):
        info_file = self.experiment.dataset.path('test_info.txt')
        index_file = self.experiment.dataset.path('test_indices.txt')
      
        if not (os.path.exists(info_file) and os.path.exists(index_file)):
            return None

        f = open(index_file)
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        indices = np.array([int(l) for l in lines])
        f.close()

        f = open(info_file)
        lines = f.readlines()
        lines = [l.strip().split(' ') for l in lines]
        video_ids = np.array([int(l[0]) for l in lines])
        video_ids = video_ids[indices]
        frames = np.array([int(l[1]) for l in lines])
        frames = frames[indices]
        track_ids = np.array([int(l[2])for l in lines])
        track_ids = track_ids[indices]
        bb_ids = np.array([int(l[3]) for l in lines])
        bb_ids = bb_ids[indices]
        f.close()

        y_true_track = []
        y_pred_track = []
        unique_video_ids = np.unique(video_ids)
        for vid_id in unique_video_ids:
            unique_track_ids = np.unique(track_ids[video_ids==vid_id])
            for track_id in unique_track_ids:
                y_true1 = y_true[np.logical_and(video_ids == vid_id, track_ids == track_id)]
                y_true_track.append(mode(y_true1)[0] * np.ones(y_true1.shape[0]))
                y_pred1 = y_pred[np.logical_and(video_ids == vid_id, track_ids == track_id)]
                y_pred_track.append(mode(y_pred1)[0] * np.ones(y_pred1.shape[0]))
        
        y_true_track = np.hstack(y_true_track)
        y_pred_track = np.hstack(y_pred_track)
        return y_pred_track, y_true_track
        

    def generate_confusion_matrix(self, y_pred, y_true):
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        scores = metrics.precision_recall_fscore_support(y_true, y_pred)
        return confusion_matrix, scores
    
                    
    def plot_confusion_matrix(self, confusion_matrix):
        counts = confusion_matrix    
        if counts is None:
            return None
 
        counts_normalised = counts.astype('float') / counts.sum(axis=1)[:, np.newaxis]
        names = self.class_labels
        
        colormap = [
        "#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
        "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"
        ]

        xname = []
        yname = []
        color = []
        alpha = []
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                xname.append(n1)
                yname.append(n2)
                
                a = np.log10(counts[j, i]+1)/np.log10(np.max(counts)+1)
                a = counts_normalised[j,i] * .9 + 0.1
                #color.append("#%02x%02x%02x" % (255, 255 - counts_normalised[j,i] * 255.0, 255 - (count / max_count) * 255.0))
                alpha.append(a)
                color.append(colormap[8])


        source = ColumnDataSource(
            data=dict(
                xname=xname,
                yname=yname,
                alphas=alpha,
                colors=color,
                count=counts.T.flatten(),
                count_normalised=counts_normalised.T.flatten() * 100.
            )
        )

        p = figure(title="Confusion Matrix",
            x_axis_location="above", tools="pan,resize,box_zoom,hover,save,tap,reset",
            x_range=names, y_range=list(reversed(names)))
        p.plot_width = 500
        p.plot_height = 500

        p.rect('xname', 'yname', 0.9, 0.9, source=source,
             color='colors', alpha='alphas', line_color=None)

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "15pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = np.pi/3
        p.xaxis.axis_label = "Predicted Class"
        p.yaxis.axis_label = "Actual Class"

        hover = p.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ('classes', '@yname, @xname'),
            ('count', '@count'),
            ('percentage', '@count_normalised'),
        ])
        
        url = "/trials/images/classification/" + self.id() + "/results/@yname"
        taptool = p.select(type=TapTool)
        taptool.action=OpenURL(url=url)

        script, div = components(p)
        return (script, div)

    def plot_scores2(self):
        
        if self.scores is None:
            return None
        
        scores = np.vstack(self.scores)
        data = OrderedDict()
        for i in xrange(len(self.class_labels)):
            data[self.class_labels[i]] = scores[:2, i]
        s1 = Bar(data, cat=['precision', 'recall'], title="Per Class Scores",
        width=500, height=500, legend=True, tools="pan,resize,box_zoom,hover,save,reset")    
        
        data = OrderedDict()
        for i in xrange(len(self.class_labels)):
            data[self.class_labels[i]] = scores[3:4, i]
        s2 = Bar(data, cat=['support'], title="Support",
        width=500, height=500, legend=True, tools="pan,resize,box_zoom,hover,save,reset")   
        
        p = hplot(s1,s2)
        
        script, div = components(p)
        return (script, div)    
        
        
    def plot_scores(self, all_scores):
        
        if all_scores is None:
            return None
        
        scores = all_scores[0][:]
        scores = np.hstack((scores,np.mean(scores)))
        class_labels = self.class_labels[:]
        class_labels.append('average')
        data = {"precision": scores}
        s1 = Bar(data, cat=class_labels, title="Per Class Precision",
        xlabel='categories', ylabel='precision', width=500, height=500,
        tools="pan,resize,box_zoom,hover,save,reset", stacked=True, palette=["#b2df8a"])    
        
        hover = s1.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
                         ('precision', '@precision'),
                          ])
        
        scores = all_scores[1][:]
        scores = np.hstack((scores,np.mean(scores)))
        class_labels = self.class_labels[:]
        class_labels.append('average')
        data = {"recall": scores}
        s2 = Bar(data, cat=class_labels, title="Per Class Recall",
        xlabel='categories', ylabel='recall', width=500, height=500,
        tools="pan,resize,box_zoom,hover,save,reset", stacked=True, palette=["#a6cee3"])  
        
        hover = s2.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
                         ('recall', '@recall'),
                          ])
        
        data = {"support": all_scores[3]}
        s3 = Bar(data, cat=self.class_labels, title="Per Class Support",
        xlabel='categories', ylabel='support', width=500, height=500,
        tools="pan,resize,box_zoom,hover,save,reset", stacked=True, palette=["#6a3d9a"])    
        
        hover = s3.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
                         ('support', '@support'),
                          ])
                          
        p = hplot(s1, s2, s3)
        
     
        
        script, div = components(p)
        return (script, div)
    
    def calculate_average_accuracy(self, confusion_matrix):
        if confusion_matrix is None:
            return None

        scores = []
        n = np.shape(confusion_matrix)[0]
        mean_acc = 0
        for index, r in enumerate(confusion_matrix):
            ss = sum(r)
            if ss != 0:
                scores.append(float(r[index]) / ss)
            

        scores = np.hstack((scores,np.mean(scores)))
        class_labels = self.class_labels[:]
        class_labels.append('average')
        data = {"accuracy": scores}
        s = Bar(data, cat=class_labels, title="Per Class Accuracy",
        xlabel='categories', ylabel='accuracy', width=500, height=500,
        tools="pan,resize,box_zoom,hover,save,reset", stacked=True, palette=["#ec5d5e"])  
        
        hover = s.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
                         ('accuracy', '@accuracy'),
                          ])

        p = hplot(s)
        script, div = components(p)
        return (script, div)


