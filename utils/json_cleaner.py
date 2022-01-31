'''
JSON cleaner - removes unwanted classes
'''

import os
import json
import normalize_classname

ROOT_DIR = '../Data/Labels/' # root directory where all JSONs are contained
SUB_DIRS = ['H6'] # subdirs containing JSONs (remove 'Labeled' from dir name)

for dir in SUB_DIRS:
    dir_path = ROOT_DIR + 'Labeled ' + dir + '/'
    for ann_path in os.listdir(dir_path):
        ann_path = ROOT_DIR + 'Labeled ' + dir + '/' + ann_path
        print(ann_path)
        with open(ann_path, 'r') as f_ann: # read JSON
            annotation_json = json.load(f_ann)

        [annotation_json['shapes'].pop(annotation_json['shapes'].index(shape)) for shape in annotation_json['shapes']
         if normalize_classname.normalize_classname(shape['label']) == 'gas entrapment porosity']

        with open(ann_path, 'w') as f_ann: # write back to the JSON
            json.dump(annotation_json, f_ann, indent=2)