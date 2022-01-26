import json
import os
from utils.model.normalize_classname import normalize_classname
import csv

csv_rows = ['Dataset', '# of images', 'lack of fusion porosity', 'gas entrapment porosity', 'keyhole porosity', 'other', 'total annotation instances']
csv_data = []

BASE_IMAGES_DIR = '../Data/Images/' # directory where all images can be found
BASE_ANNOTATIONS_DIR = '../Data/Labels/' # directory where all images labels can be found
IMAGES_DIRS = ['H6', 'J8', 'J0', 'J4', 'R6', 'R0']

for i in range(len(IMAGES_DIRS)):
    row = []
    image_dir = BASE_IMAGES_DIR + IMAGES_DIRS[i] + '/'
    annotation_dir = BASE_ANNOTATIONS_DIR + 'Labeled ' + IMAGES_DIRS[i] + '/'
    row.append(IMAGES_DIRS[i])
    row.append(str(len(os.listdir(image_dir))))

    classes = {'lack of fusion porosity': 0,
     'gas entrapment porosity': 0,
     'keyhole porosity': 0,
     'other': 0}

    total_annotations = 0

    for annotation in os.listdir(annotation_dir):
        path = annotation_dir + annotation
        f_ann = open(path, )
        annotation_json = json.load(f_ann)
        for shape in annotation_json['shapes']:
            label = normalize_classname(shape['label'])
            classes[label] += 1

    for label in classes.keys():
        total_annotations += classes.get(label)
        row.append(classes.get(label))

    row.append(total_annotations)
    csv_data.append(row)

with open('dataset_summary.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(csv_rows)
    write.writerows(csv_data)