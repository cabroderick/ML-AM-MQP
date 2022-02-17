import json

SCALE_RATIO = 2
ROOT_IMG_DIR = "../Stitched/"
sets_name = ["G9"]

for set in sets_name:
    annotation_json = json.load(open(ROOT_IMG_DIR+set+"_merged_50.json"))
    shapes = []
    for label in annotation_json["shapes"]:
        label["points"] = [[SCALE_RATIO*t[0], SCALE_RATIO*t[1]] for t in label["points"]]
        shapes.append(label)
    annotation_json["shapes"] = shapes

    with open(ROOT_IMG_DIR+set+"_merged_scaled.json", 'w') as outfile:
        json.dump(annotation_json, outfile)
