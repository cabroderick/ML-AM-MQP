'''
JSON cleaner - removes unwanted classes
Replace SUB_DIRS with the appropriate annotation directories and you're set!
'''

import os
import json
from normalize_classname import normalize_classname
from load_dataset_paths import load_dataset_paths
import cv2
import numpy as np

def normalize_dimensions(col_min, col_max, row_min, row_max):
    return max(col_min, 0), col_max, max(row_min, 0), row_max

ROOT_IMG_DIR = '/home/cabroderick/Data/Images/' # root directory where all JSONs are contained
ROOT_ANN_DIR = '/home/cabroderick/Data/Labels/'
SUB_DIRS = ['G0',
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
THRESHOLD_1 = 2492.50
THRESHOLD_2 = 14661.50

img_dirs, ann_dirs = load_dataset_paths(ROOT_IMG_DIR, ROOT_ANN_DIR, SUB_DIRS)

for i in range(len(ann_dirs)):
    ann_path = ann_dirs[i]
    img_path = img_dirs[i]
    print(ann_path)
    with open(ann_path, 'r') as f_ann:  # read JSON
        annotation_json = json.load(f_ann)

    image = cv2.imread(img_path)

    shapes = []
    for shape in annotation_json['shapes']:
        shape['label'] = normalize_classname(shape['label'])
        if shape['label'] != 'gas entrapment porosity' and shape['label'] != 'other':
            shape_type = ''
            if 'shape_type' in shape:
                shape_type = shape['shape_type']
            else:
                points = []
                for p in shape['points']:
                    points.append(p[0])
                    points.append(p[1])
                if len(points) == 4:
                    shape_type = 'rectangle'
                else:
                    shape_type = 'polygon'

            # print(shape_type)

            # exit(0)
            if shape_type == 'rectangle':
                # extract row and col data and crop image to annotation size
                col_min, col_max = int(min(shape['points'][0][0], shape['points'][1][0])), int(
                    max(shape['points'][0][0], shape['points'][1][0]))
                row_min, row_max = int(min(shape['points'][0][1], shape['points'][1][1])), int(
                    max(shape['points'][0][1], shape['points'][1][1]))
                col_min, col_max, row_min, row_max = normalize_dimensions(col_min, col_max, row_min, row_max)
                cropped_img = image[row_min:row_max, col_min:col_max]  # crop image to size of bounding box
                cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                edged = cv2.Canny(cropped_img_gray, 30, 200)

                # apply contour to image and fill
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                dilated = cv2.dilate(edged, kernel)
                contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                total_area = sum([cv2.contourArea(c) for c in contours])

            elif shape_type == 'polygon':
                # generate mask from polygon points
                points = []
                [points.append(coord) for coord in shape['points']]
                points = np.array(points, dtype=np.int32)
                polygon_mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillPoly(polygon_mask, [points], (255, 255, 255))

                # apply mask
                cropped_img = cv2.bitwise_and(image, polygon_mask)
                black_pixels = np.where(
                    (cropped_img[:, :, 0] == 0) &
                    (cropped_img[:, :, 1] == 0) &
                    (cropped_img[:, :, 2] == 0)
                )

                cropped_img[black_pixels] = (0, 255, 255)

                cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                edged = cv2.Canny(cropped_img_gray, 30, 200)

                # apply contour to image and fill
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                dilated = cv2.dilate(edged, kernel)
                contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                total_area = sum([cv2.contourArea(c) for c in contours])

            if total_area > 800: # thresholding
                if shape['label'] == 'lack of fusion porosity':
                    if total_area < THRESHOLD_1:
                        shape["label"] = "small lack of fusion porosity"
                    elif total_area < THRESHOLD_2:
                        shape["label"] = "medium lack of fusion porosity"
                    else:
                        shape["label"] = "large lack of fusion porosity"
            
                shapes.append(shape)
            

    os.remove(ann_path)  # delete old ann path

    if shapes:  # only generate new JSON if there are any annotations left
        if ann_path[-12:].replace('-', '_').lower() == '_20x_yz.json':
            print(ann_path[-12:])
            print('Removing 20X_YZ')
            ann_path = ann_path[:-12] + ann_path[-5:]  # remove 20X_YZ at the end of annotations

        annotation_json['shapes'] = shapes
        with open(ann_path, 'w') as f_ann:  # write back to the JSON
            json.dump(annotation_json, f_ann, indent=2)
    else:
        print('No annotations found - removing image and json')
        os.remove(img_path) # remove image associated with json if it has no valid labels