'''
Creates summary of dataset by class, files, etc.
Usage: python dataset_summary.py [optional output filename]
'''

import json
import os
import sys
from normalize_classnames import normalize_classname

if len(sys.argv) > 1:
    sys.stdout = open(sys.argv[1], 'w')

BASE_IMAGES_DIR = './Data/Images/' # directory where all images can be found
BASE_ANNOTATIONS_DIR = './Data/Labels/' # directory where all images labels can be found
IMAGES_DIRS = ['G0/', 'G9/', 'H0/', 'H4/', 'H5/', 'H6/', 'H8/', 'H9/', 'J0/', 'J1/', 'J3/', 'J4/', 'J7/',
                 'J8/', 'K0/', 'Q0/', 'Q3/', 'Q5/', 'Q9/', 'R2/', 'R6/', 'R7/'] # list of directories where images are contained
ANNOTATIONS_DIRS = ['Labeled G0/', 'Labeled G9/', 'Labeled H0/', 'Labeled H4/', 'Labeled H5/', 'Labeled H6/',
                      'Labeled H8/', 'Labeled H9/', 'Labeled J0/', 'Labeled J1/', 'Labeled J3/', 'Labeled J4/',
                      'Labeled J7/', 'Labeled J8/', 'Labeled K0/', 'Labeled Q0/', 'Labeled Q3/', 'Labeled Q5/',
                      'Labeled Q9/', 'Labeled R2/', 'Labeled R6/', 'Labeled R7/'] # corresponding list of directories where annotations are contained

classes_all = {}

for i in range(len(IMAGES_DIRS)):
    image_dir = BASE_IMAGES_DIR + IMAGES_DIRS[i]
    annotation_dir = BASE_ANNOTATIONS_DIR + ANNOTATIONS_DIRS[i]
    print('Dataset: ' + IMAGES_DIRS[i][:-1])
    print('Total images: ' + str(len(os.listdir(image_dir))))

    classes = {} # dictionary containing the classes for a particular dataset

    for annotation in os.listdir(annotation_dir):
        path = annotation_dir + annotation
        f_ann = open(path, )
        annotation_json = json.load(f_ann)
        for shape in annotation_json['shapes']:
            label = normalize_classname(shape['label'])
            if label in classes.keys():
                classes[label] += 1
            else:
                classes[label] = 1

    print('Number of annotation classes: ' + str(len(classes.keys())))
    print('Annotation occurrences:')
    for label in classes.keys():
        print(label + ': ' + str(classes.get(label)) + ' occurrences')
    print('=========================================')

    for key in classes.keys():
        if key in classes_all.keys():
            classes_all[key] += classes.get(key)
        else:
            classes_all[key] = classes.get(key)

print('Total annotation occurrences:')
for label in classes_all.keys():
    print(label + ': ' + str(classes_all.get(label)) + ' occurrences')

if len(sys.argv) > 1:
    sys.stdout.close()