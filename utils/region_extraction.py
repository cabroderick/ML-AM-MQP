import json
from shapely.geometry.polygon import Polygon
from shapely.geometry import box
import cv2

SCALE_RATIO = 2
ROOT_IMG_DIR = "../Stitched/"
IMG_OUT_DIR = '../Regions/Images/'
LABELS_OUT_DIR = '../Regions/Labels/'
sets_name = ["G9/"]

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
        col_min, col_max = int(min(r['points'][0][0], r['points'][1][0])), int(max(r['points'][0][0], r['points'][1][0]))
        row_min, row_max = int(min(r['points'][0][1], r['points'][1][1])), int(max(r['points'][0][1], r['points'][1][1]))
        cropped_img = img[row_min:row_max, col_min:col_max]
        region = box(r["points"][0][0], r["points"][0][1], r["points"][1][0], r["points"][1][1])
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
