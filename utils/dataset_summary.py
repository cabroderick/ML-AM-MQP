import json
import os
from utils.normalize_classnames import normalize_classname
import csv

csv_rows = ['Dataset', '# of images', 'lack of fusion porosity', 'gas entrapment porosity', 'keyhole porosity', 'other', 'total annotation instances']
csv_data = []

BASE_IMAGES_DIR = './Data/Images/' # directory where all images can be found
BASE_ANNOTATIONS_DIR = './Data/Labels/' # directory where all images labels can be found
IMAGES_DIRS = ['G0/', 'G3/', 'G9/', 'H0/', 'H4/', 'H5/', 'H6/', 'H8/', 'H9/', 'J0/', 'J1/', 'J3/', 'J4/', 'J7/',
                 'J8/', 'K0/', 'K4/', 'Q0/', 'Q3/', 'Q5/', 'Q9/', 'R0/', 'R2/', 'R6/', 'R7/']
# corresponding list of directories where annotations are contained
ANNOTATIONS_DIRS = ['Labeled G0/', 'Labeled G3/', 'Labeled G9/', 'Labeled H0/', 'Labeled H4/', 'Labeled H5/', 'Labeled H6/',
                      'Labeled H8/', 'Labeled H9/', 'Labeled J0/', 'Labeled J1/', 'Labeled J3/', 'Labeled J4/',
                      'Labeled J7/', 'Labeled J8/', 'Labeled K0/', 'Labeled K4/', 'Labeled Q0/', 'Labeled Q3/', 'Labeled Q5/',
                      'Labeled Q9/', 'Labeled R0/', 'Labeled R2/', 'Labeled R6/', 'Labeled R7/']

for i in range(len(IMAGES_DIRS)):
    row = []
    image_dir = BASE_IMAGES_DIR + IMAGES_DIRS[i]
    annotation_dir = BASE_ANNOTATIONS_DIR + ANNOTATIONS_DIRS[i]
    row.append(IMAGES_DIRS[i][:-1])
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