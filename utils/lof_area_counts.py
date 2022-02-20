from load_dataset_paths import load_dataset_paths
import json

ROOT_IMG_DIR = '/home/cabroderick/Data/Images/' # root directory where all JSONs are contained
ROOT_ANN_DIR = '/home/cabroderick/Data/Labels/'
DIRS = ['G0',
'G7',
'G8',
'G9',
'H0',
'H4',
'H5',
'H6',
'H7',
'H8',
'H9',
'J0',
'J1',
'J3',
'J4',
'J4R',
'J5',
'J7',
'J8',
'J9',
'K0',
'K0R',
'K1',
'K4',
'K5',
'Q0',
'Q3',
'Q4',
'Q5',
'Q6',
'Q9',
'R0',
'R2',
'R5',
'R6',
'R7'] # subdirs containing JSONs (remove 'Labeled' from dir name)

counts = {'small lack of fusion porosity': 0, 'medium lack of fusion porosity': 0, 'large lack of fusion porosity': 0, 'keyhole porosity': 0}

img_dirs, ann_dirs = load_dataset_paths(ROOT_IMG_DIR, ROOT_ANN_DIR, DIRS)

for dir in ann_dirs:
    with open(dir, 'r') as f:
        data = json.load(f)
        for shape in data['shapes']:
            if shape['label'] in counts.keys():
                counts[shape['label']] += 1

print(counts)