import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT_IMG_DIR = '../Stitched/'
DIRS = ["K0", "Q0"]

data = {}


def normalize_dimensions(col_min, col_max, row_min, row_max):
    return max(col_min, 0), col_max, max(row_min, 0), row_max


for d in DIRS:
    data[d] = {'lack of fusion porosity': [], 'keyhole porosity': []}

    ann_path = ROOT_IMG_DIR + d + "_merged.json"
    image_path = ROOT_IMG_DIR + d + ".png"

    image = cv2.imread(image_path)

    with open(ann_path, 'r') as f_ann:  # read JSON
        annotation_json = json.load(f_ann)

    for a in annotation_json["shapes"]:
        if a['label'] == 'lack of fusion porosity' or a['label'] == 'keyhole porosity':
            if a['shape_type'] == 'rectangle':
                # extract row and col data and crop image to annotation size
                col_min, col_max = int(min(a['points'][0][0], a['points'][1][0])), int(
                    max(a['points'][0][0], a['points'][1][0]))
                row_min, row_max = int(min(a['points'][0][1], a['points'][1][1])), int(
                    max(a['points'][0][1], a['points'][1][1]))
                col_min, col_max, row_min, row_max = normalize_dimensions(col_min, col_max, row_min, row_max)
                cropped_img = image[row_min:row_max, col_min:col_max]  # crop image to size of bounding box
                cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                edged = cv2.Canny(cropped_img_gray, 30, 200)

                # apply contour to image and fill
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                dilated = cv2.dilate(edged, kernel)
                contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                total_area = sum([cv2.contourArea(c) for c in contours])

            elif a['shape_type'] == 'polygon':
                # generate mask from polygon points
                points = []
                [points.append(coord) for coord in a['points']]
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
            class_type = a["label"]
            data[d][class_type].append(total_area)
new_df = pd.DataFrame.from_dict(data)
new_df.to_csv("area_dist.csv")
