'''
JSON cleaner - removes unwanted classes
Replace SUB_DIRS with the appropriate annotation directories and you're set!
'''

import os
import json
import normalize_classname

ROOT_DIR = '../Data/Labels/' # root directory where all JSONs are contained
SUB_DIRS = ['J4'] # subdirs containing JSONs (remove 'Labeled' from dir name)

for dir in SUB_DIRS:
    dir_path = ROOT_DIR + 'Labeled ' + dir + '/'
    for file in os.listdir(dir_path):
        ann_path = ROOT_DIR + 'Labeled ' + dir + '/' + file
        print(ann_path)
        with open(ann_path, 'r') as f_ann: # read JSON
            annotation_json = json.load(f_ann)
        # rename filenames to remove 20X_YZ.json

        shapes = []
        for shape in annotation_json['shapes']:
            shape['label'] = normalize_classname.normalize_classname(shape['label'])
            if shape['label'] != 'gas entrapment porosity':
                shapes.append(shape)

        os.remove(ann_path) # delete old ann path
        if file[-12:] == '_20X_YZ.json':
            ann_path = ann_path[:-12] + ann_path[-5:] # remove 20X_YZ at the end of annotations

        annotation_json['shapes'] = shapes

        with open(ann_path, 'w') as f_ann: # write back to the JSON
            json.dump(annotation_json, f_ann, indent=2)