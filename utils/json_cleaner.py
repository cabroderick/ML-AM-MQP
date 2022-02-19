'''
JSON cleaner - removes unwanted classes
Replace SUB_DIRS with the appropriate annotation directories and you're set!
'''

import os
import json
from utils.normalize_classname import normalize_classname
from utils.load_dataset_paths import load_dataset_paths

ROOT_IMG_DIR = '../Data/Images/' # root directory where all JSONs are contained
ROOT_ANN_DIR = '../Data/Labels/'
SUB_DIRS = ['G7'] # subdirs containing JSONs (remove 'Labeled' from dir name)

img_dirs, ann_dirs = load_dataset_paths(ROOT_IMG_DIR, ROOT_ANN_DIR, SUB_DIRS)

for i in range(len(ann_dirs)):
    ann_path = ann_dirs[i]
    img_path = img_dirs[i]
    print(ann_path)
    with open(ann_path, 'r') as f_ann:  # read JSON
        annotation_json = json.load(f_ann)

    shapes = []
    for shape in annotation_json['shapes']:
        shape['label'] = normalize_classname(shape['label'])
        if shape['label'] != 'gas entrapment porosity' and shape['label'] != 'other':
            shapes.append(shape)

    os.remove(ann_path)  # delete old ann path

    if shapes:  # only generate new JSON if there are any annotations left
        if ann_path.replace('-', '_').lower() == '_20x_yz.json':
            ann_path = ann_path[:-12] + ann_path[-5:]  # remove 20X_YZ at the end of annotations

        annotation_json['shapes'] = shapes
        with open(ann_path, 'w') as f_ann:  # write back to the JSON
            json.dump(annotation_json, f_ann, indent=2)
    else:
        print('No annotations found - removing image and json')
        os.remove(img_path) # remove image associated with json if it has no valid labels