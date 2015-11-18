#!/usr/bin/python
# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os
import time
import argparse
import logging

try:
    import digits
except ImportError:
    # Add path for DIGITS package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import utils, log

from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np

logger = logging.getLogger('digits.tools.svm_train_test')

class SVMTask:
    """
    Runs caffe on a test dataset in lmdb format using a pretrained model
    """

    def __init__(self, 
                 YTrain_file,
                 XTrain_file,
                 XTest_file,
                 output_path,
                 normalise,
                 C,
                 class_weight,
                 ):
        """
        Arguments:
      
        """
        self.YTrain = joblib.load(YTrain_file)
        XTrain = joblib.load(XTrain_file)
        self.XTrain = XTrain.reshape(np.size(XTrain, axis=0), -1)
       
        XTest = joblib.load(XTest_file)   
        self.XTest = XTest.reshape(np.size(XTest, axis=0), -1)

        self.output_path = output_path
    
        if normalise:
            normalizer = Normalizer(copy=False)
            normalizer.transform(self.XTrain)
            normalizer.transform(self.XTest)

        self.C = C
        if class_weight == 'none':
            class_weight = None
        self.class_weight = class_weight
        
        
    def evaluate(self):
        """
        Returns True on success

        Arguments:

        Keyword arguments:
       
        """
        svm = LinearSVC(C=self.C, class_weight=self.class_weight)
        svm.fit(self.XTrain, self.YTrain)
        prob = svm.decision_function(self.XTest)
        # To make the binary case consistent with non-binary cases
        if prob.ndim == 1:
            prob = np.vstack((-prob, prob)).T
        YPred = svm.predict(self.XTest)
        joblib.dump(YPred, self.output_path + '/ypred.jbl')
        joblib.dump(prob, self.output_path + '/prob.jbl')
                    
        return True
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM tool - DiGiTS')

    ### Positional arguments

    parser.add_argument('YTrain_file',
            help='')
    parser.add_argument('XTrain_file',
            help='')
    parser.add_argument('XTest_file',
            help='',
            )
    parser.add_argument('output_path',
            help='',
            )
    parser.add_argument('normalise',
            help='',
            type=int,
            )
    parser.add_argument('C',
            help='',
            type=float,
            )
    parser.add_argument('class_weight',
            help='',
            )
    args = vars(parser.parse_args())
    svm_task = SVMTask(args['YTrain_file'],
                       args['XTrain_file'],
                       args['XTest_file'],
                       args['output_path'],
                       args['normalise'],
                       args['C'],
                       args['class_weight'],
                       )

    if svm_task.evaluate():
        sys.exit(0)
    else:
        logger.error('Exiting with Error')
        sys.exit(1)

