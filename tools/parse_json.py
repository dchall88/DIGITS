#!/usr/bin/python
# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os
import re
import argparse
import time
import logging
from random import shuffle
from urlparse import urlparse
import urllib
import pickle
import json

import requests

try:
    import digits
except ImportError:
    # Add path for DIGITS package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import utils, log

logger = logging.getLogger('digits.tools.parse_json')


def validate_output_file(filename):
    if filename is None:
        return True
    if os.path.exists(filename):
        logger.error('output file "%s" already exists!' % filename)
        return False
    output_dir = os.path.dirname(filename)
    if not output_dir:
        output_dir = '.'
    if not os.path.exists(output_dir):
        logger.error('output directory "%s" does not exist!' % output_dir)
        return False
    if not os.access(output_dir, os.W_OK):
        logger.error('you do not have write access to output directory "%s"!' % output_dir)
        return False
    return True

def validate_input_file(filename):
    if not os.path.exists(filename) or not os.path.isfile(filename):
        logger.error('input file "%s" does not exist!' % filename)
        return False
    if not os.access(filename, os.R_OK):
        logger.error('you do not have read access to "%s"!' % filename)
        return False
    return True

def validate_range(number, min=None, max=None, allow_none=False):
    if number is None:
        if allow_none:
            return True
        else:
            logger.error('invalid value %s' % number)
            return False
    try:
        float(number)
    except ValueError:
        logger.error('invalid value %s' % number)
        return False

    if min is not None and number < min:
        logger.error('invalid value %s' % number)
        return False
    if max is not None and number > max:
        logger.error('invalid value %s' % number)
        return False
    return True


def parse_json(json_annotation_file,
               train_test_split_file,
               labels_file,
               get_bboxes = False,
               warp_bboxes = False,
               train_file=None,
               val_file=None,
               test_file=None,
               min_per_category=2,
               max_per_category=None,
               ):
    """
    Parses a json annotation file into three textfiles - train, val and test -
    given the videos to split on.
    Returns True on success

    Arguments:
    json_annotation_file -- a file containing annotations and image locations for a dataset
    train_test_split_file -- a file containing the videos to use for each set
    labels_file -- file for labels

    Keyword Arguments:
    get_bboxes -- whether to parse using bounding boxes in the images
    train_file -- output file for training images
    val_file -- output file for validation images
    test_file -- output file for test images
    min_per_category -- minimum number of images per category
    max_per_category -- maximum number of images per category
    """
    ## TODO - Add get bounding boxes functionality

    ## TODO - ADD FILE VALIDATION
    ### Read the splits from train_test_split_file
    splits = []
    with open(train_test_split_file) as infile:
        for line in infile:
            line = line.strip()
            splits.append(line)

    if len(splits) < 3:
        logger.error('the train_test_split file must contain 3 lines')
        return False

    train_splits = splits[0].split()
    val_splits = splits[1].split()
    test_splits = splits[2].split()

    ## TODO - ADD JSON OR PKL FORMAT VALIDATION
    ### Load Annotation File
    with open(json_annotation_file, 'r') as f:
        if json_annotation_file[-3:] == 'pkl':
            dataset = pickle.load(f)
        else:
            dataset = json.load(f)
    
    train_count, subcategories, classes = parse_json_set(dataset, train_splits, train_file, labels_file=labels_file)
    val_count = parse_json_set(dataset, val_splits, val_file)
    test_count = parse_json_set(dataset, test_splits, test_file)

    logger.info('Found %d images.' % (train_count + val_count + test_count, ))
    for subcategory, class1 in zip(subcategories, classes):
        logger.info('Found %d classes in the %s subcategory.' % (len(class1), subcategory))
    logger.info('Selected %d for training.' % train_count)
    logger.info('Selected %d for validation.' % val_count)
    logger.info('Selected %d for testing.' % test_count)
    return True


def parse_json_set(dataset, split, output_file, labels_file=None):
    
    outfile = open(output_file, 'w')
    infofile = open(output_file[:-4] + '_info.txt', 'w')
     
    lines = []
    lines_info = []

    for video_index, video_name in enumerate(split):

        images = dataset[video_name]['images']

        attributes = dataset[video_name]['attributes']
        attribute_names = attributes['names'].keys()
        attribute_classes = attributes['names'].values()
        tracks = dataset[video_name]['tracks']

        cnt = 0
        for track_id in tracks:
            if track_id != 0:
                track = tracks[track_id]
                attribute = attributes['values'][track_id]
                attribute = [str(x) for x in attribute.values()]
                attribute = ' '.join(attribute)
                #attribute = attribute['sex']


                for annotation in track:
                    bbox = annotation['bounding box']
                    x = int(bbox['x'])
                    y = int(bbox['y'])
                    w = int(bbox['width'])
                    h = int(bbox['height'])

                    pose = annotation['pose']
                    present = [xx[2] for xx in pose.values()]

                    if present.count(-1) == 0:

                        # Normalise
                        bb = [x, y, w, h]

                        frame = annotation['frame']
                        bb_id = annotation['bounding_box_id']
                        image_path = images['path'][frame]

                        bb = [str(b1) for b1 in bb]
                        bb = ' '.join(bb)

                        lines.append('%s %s %s' % (image_path, bb, attribute))
                        lines_info.append('%s %s %s %s' % (video_index, frame, track_id, bb_id))
                        
        logger.debug('Progress: %0.2f' % (float(video_index)/len(split)))

    outfile.write('\n'.join(lines) + '\n')
    outfile.close()
    
    infofile.write('\n'.join(lines_info) + '\n')
    infofile.close()
    
    if labels_file:
        with open(labels_file, 'w') as labels_outfile:
            for category_type, class_labels in attributes['names'].items():
                class_labels1 = [cl.replace(' ', '_') for cl in class_labels]
                labels_outfile.write(category_type + ' ' + ' '.join(class_labels1) + '\n')
        return len(lines), attribute_names, attribute_classes
    else:
        return len(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse-JSON Annotation tool - DiGiTS')

    ### Positional arguments

    parser.add_argument('json_annotation_file',
            help='A filesystem path to the json annotation file - can be a pickle file also'
            )
    parser.add_argument('train_test_split_file',
            help='The file containing the videos to use in the dataset for training, validation and testing'
            )
    parser.add_argument('labels_file',
            help='The file containing labels.'
            )
    ### Optional arguments
    parser.add_argument('-bb', '--get_bboxes',
            type=bool,
            help='Extract bounding boxes from images'
            )
    parser.add_argument('-wbb', '--warp_bboxes',
            type=bool,
            help='Warp the bounding boxes'
            )
    parser.add_argument('-t', '--train_file',
            help='The output file for training images'
            )
    parser.add_argument('-v', '--val_file',
            help='The output file for validation images'
            )
    parser.add_argument('-s', '--test_file',
            help='The output file for test images'
            )
    parser.add_argument('--min',
            type=int,
            metavar='MIN_PER_CATEGORY',
            default=1,
            help="What is the minimum allowable number of images per category? (categories which don't meet this criteria will be ignored) [default=2]"
            )
    parser.add_argument('--max',
            type=int,
            metavar='MAX_PER_CATEGORY',
            help='What is the maximum limit of images per category? (categories which exceed this limit will be trimmed down) [default=None]'
            )

    args = vars(parser.parse_args())

    for valid in [
            validate_input_file(args['json_annotation_file']),
            validate_input_file(args['train_test_split_file']),
            validate_output_file(args['train_file']),
            validate_output_file(args['val_file']),
            validate_output_file(args['test_file']),
            validate_range(args['min'], min=1),
            validate_range(args['max'], min=1, allow_none=True),
            ]:
        if not valid:
            sys.exit(1)

    start_time = time.time()

    if parse_json(args['json_annotation_file'],
                    args['train_test_split_file'],
                    args['labels_file'],
                    train_file=args['train_file'],
                    val_file=args['val_file'],
                    test_file=args['test_file'],
                    min_per_category=args['min'],
                    max_per_category=args['max'],
                    ):
        logger.info('Done after %d seconds.' % (time.time() - start_time))
        sys.exit(0)
    else:
        sys.exit(1)

