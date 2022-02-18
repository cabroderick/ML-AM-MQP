import json
from shapely.geometry.polygon import Polygon
from shapely.geometry import box
import cv2
import numpy as np

SCALE_RATIO = 2
ROOT_IMG_DIR = "../Stitched/"
IMG_OUT_DIR = '../Regions/Images/'
LABELS_OUT_DIR = '../Regions/Labels/'
sets_name = ["Q0_50/"]

for set in sets_name:
    annotation_json = json.load(open(ROOT_IMG_DIR+set[:-1]+"_merged_regions.json"))
    img = cv2.imread(ROOT_IMG_DIR+set[:-1] + ".png")
    shapes = []
    regions = []
    instances = []
    for label in annotation_json["shapes"]:
        if label["label"] == "Region":
            regions.append(label)
        else:
            instances.append(label)

    for i in range(len(regions)):
        new_annot = {"shapes": []}
        r = regions[i]
        if r['shape_type'] == 'rectangle':
            col_min, col_max = int(min(r['points'][0][0], r['points'][1][0])), int(
                max(r['points'][0][0], r['points'][1][0]))
            row_min, row_max = int(min(r['points'][0][1], r['points'][1][1])), int(
                max(r['points'][0][1], r['points'][1][1]))
            cropped_img = img[row_min:row_max, col_min:col_max]
            region = box(r["points"][0][0], r["points"][0][1], r["points"][1][0], r["points"][1][1])
        elif r['shape_type'] == 'polygon':
            points = []
            [points.append(coord) for coord in r['points']]
            x_coords = [point[0] for point in points]
            col_min, col_max = int(min(x_coords)), int(max(x_coords))
            y_coords = [point[1] for point in points]
            row_min, row_max = int(min(y_coords)), int(max(y_coords))

            cropped_img = img[row_min:row_max, col_min:col_max]
            points = [(point[0] - col_min, point[1] - row_min) for point in points] # adjust coords of points
            points = np.array(points, dtype=np.int32)

            polygon_mask = np.zeros(cropped_img.shape, dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [points], (255, 255, 255))

            # apply mask
            cropped_img = cv2.bitwise_and(cropped_img, polygon_mask)
            black_pixels = np.where(
                (cropped_img[:, :, 0] == 0) &
                (cropped_img[:, :, 1] == 0) &
                (cropped_img[:, :, 2] == 0)
            )
            cropped_img[black_pixels] = (0, 0, 0)

        shapes = []
        for l in instances:
            if l["shape_type"] == "polygon":
                instance = Polygon(l["points"])
            else:
                instance = box(l["points"][0][0], l["points"][0][1], l["points"][1][0], l["points"][1][1])
            if instance.intersects(region):
                new_ins = [[p[0]-col_min, p[1]-row_min] for p in l["points"]]
                new_label = {}
                new_label["points"] = new_ins
                new_label["label"] = l["label"]
                shapes.append(new_label)
        new_annot["shapes"] = shapes
        cv2.imwrite(IMG_OUT_DIR+set+set[:-1]+'_'+ str(i)+'.png', cropped_img)

        with open(LABELS_OUT_DIR+set+set[:-1]+'_'+ str(i)+".json", 'w') as outfile:
            json.dump(new_annot, outfile)
