#!/usr/bin/python
# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os
import time
import argparse
import logging
from re import match as re_match
from shutil import rmtree
import random
import threading
import Queue
from sklearn.externals import joblib

try:
    import digits
except ImportError:
    # Add path for DIGITS package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import utils, log

import numpy as np
import pickle
from cStringIO import StringIO
# must import digits.config before caffe to set the path
import caffe
from caffe.proto import caffe_pb2

logger = logging.getLogger('digits.tools.caffe_extract')

class CaffeExtract:
    """
    Runs caffe on a test dataset in lmdb format using a pretrained model
    """

    def __init__(self, 
                 batch_deploy_file, 
                 caffe_model, 
                 layers,
                 output_path, 
                 num_test_examples, 
                 ):
        """
        Arguments:
        batch_deploy_file -- prototxt file containing information about dataset to test on
        caffe_model -- caffemodel file to use on the dataset
        """
        ### Validate input
        logger.error('batch_deploy_file: %s, does not exist' % batch_deploy_file)
        logger.error('caffe model: %s,  does not exist' % caffe_model)
        if not os.path.exists(batch_deploy_file):
            logger.error('batch_deploy_file: %s, does not exist' % batch_deploy_file)
            return False
        
        if not os.path.exists(caffe_model):
            logger.error('caffe model: %s,  does not exist' % caffe_model)
            return False
        
        self.layers = layers.split()
        self.batch_deploy_file = batch_deploy_file
        self.caffe_model = caffe_model
        self.output_path = os.path.dirname(os.path.join(output_path, ''))
        self.num_test_examples = num_test_examples

    def evaluate_model(self):
        """
        Returns True on success

        Arguments:

        Keyword arguments:
       
        """
        if digits.config.config_value('caffe_root')['cuda_enabled'] and\
            digits.config.config_value('gpu_list'):
            caffe.set_mode_gpu()

        net = caffe.Net(self.batch_deploy_file, self.caffe_model, caffe.TEST)
        if not set(self.layers) <= set(net.blobs.keys()):
            logger.error('attempting to extract data from a non-existent layer')
            logger.error('%s, %s' % (self.layers, type(self.layers)))
            logger.error('%s' % net.blobs.keys())
            return False        
        logger.error('%s, %s' % (self.layers, type(self.layers)))
        
        ### Start working
        start = time.time()
        
        batch_size = net.blobs['data'].num
        
        output = {}       
        out = net.forward()
        for layer in self.layers:
            output[layer] = net.blobs[layer].data.copy()
                        
        for x in xrange(batch_size, self.num_test_examples, batch_size):
            logger.error('%s, %s' % (batch_size, self.num_test_examples))
            out = net.forward()
            for layer in self.layers:
                logger.error('%s. %s' % (layer, np.shape(net.blobs[layer].data)))
                if net.blobs[layer].data.ndim == 1:
                    output[layer] = np.hstack((output[layer], net.blobs[layer].data))
                else:
                    output[layer] = np.vstack((output[layer], net.blobs[layer].data))
        
        # Tidy up (remove padding from output if num_examples is not a multiple of batch_size
        for layer in self.layers:
            output[layer] = np.squeeze(output[layer][:self.num_test_examples])

        logger.info('Evaluation completed after %d seconds.' % (time.time() - start))
        
        ### Save output
        for layer, ftrs in output.items():
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
            layer1 = layer.replace('/', '_')
            joblib.dump(ftrs, os.path.join(self.output_path, layer1 + '.jbl')) 

        return True
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Caffe Extract tool - DiGiTS')

    ### Positional arguments

    parser.add_argument('batch_deploy_file',
            help='An input prototxt file with DATA layer defining test set')
    parser.add_argument('caffe_model',
            help='caffe model to on which to evaluate the test set')
    parser.add_argument('layers',
            help='from what layers in the model to extract values',
            )
    parser.add_argument('output_path',
            help='where to save the results'
            )
    parser.add_argument('num_test_examples',
            type=int,
            help='total number of test examples'
            )
    
    args = vars(parser.parse_args())
    logger.error('Exiting with Error')
    caffe_extract = CaffeExtract(args['batch_deploy_file'],
                                 args['caffe_model'],
                                 args['layers'],
                                 args['output_path'],
                                 args['num_test_examples'],)

    if caffe_extract.evaluate_model():
        sys.exit(0)
    else:
        logger.error('Exiting with Error')
        sys.exit(1)

