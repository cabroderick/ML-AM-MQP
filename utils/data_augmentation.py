import os
import sys
import json
import imageio
import imgaug


from utils.dataset import Model_Dataset

if len(sys.argv) < 2: # ensure model name is included in arguments
    print("Usage: python data_augmentation.py <augmentation_method> [image directories file] [path to save images]")

aug_method = sys.argv[1]
dataset = Model_Dataset()

sets_list = dataset.IMG_DIRS

if len(sys.argv) > 2:
    paths_file = open(sys.argv[2], 'r')
    img_dirs = paths_file.readlines()
    sets_list = img_dirs

for set in sets_list:
    for filename in os.listdir(dataset.ROOT_IMG_DIR + set):
        image = imageio.imread(dataset.ROOT_IMG_DIR + set+"\\"+filename)

        if aug_method == "gaussian_blur":
            aug = imgaug.augmenters.GaussianBlur(sigma=(0.0, 3.0))
        elif aug_method == "fliplr":
            aug = imgaug.augmenters.Fliplr(1)
        elif aug_method == "flipud":
            aug = imgaug.augmenters.Flipud(1)

        img_aug = aug(image=image)
        annotations = {}
        annotations['shapes'] = []
        new_label = {}
        new_label["group_id"] = None
        new_label["flags"] = []
        new_label["shape_type"] = ""
        new_label["label"] = ""

        labels_file_path = dataset.ROOT_ANNOTATION_DIR + "Labeled " + set +"\\"+ filename[:-4] + ".json"
        if os.path.exists(labels_file_path):
            f_ann = open(labels_file_path, )
        else:
            labels_file_path = dataset.ROOT_ANNOTATION_DIR+ "Labeled " + set + filename[:-4] + "_20X_YZ.json"
            f_ann = open(labels_file_path, )
        annotation_json = json.load(f_ann)
        for label in annotation_json["shapes"]:
            this_label = new_label
            this_label["shape_type"] = label["shape_type"]
            this_label["label"] = label["label"]

            if  this_label["shape_type"] == "rectangle":
                x1, y1, x2, y2 =  label["points"][0][0], label["points"][0][1], label["points"][1][0], \
                                  label["points"][1][1]
                _, aug_shape = aug(image=image,bounding_boxes=imgaug.augmentables.bbs.BoundingBox(x1, y1, x2, y2))
                this_label["points"] = aug_shape.coords.tolist()
            elif this_label["shape_type"] == "polygon":
                _, aug_shape = aug(image=image, polygons=imgaug.augmentables.polys.Polygon(label["points"]))
                this_label["points"] = aug_shape.coords.tolist()
            print(this_label)
            annotations["shapes"].append(this_label)

        #save aug annotations to a file
        with open(dataset.ROOT_ANNOTATION_DIR + set+"_"+aug_method+"//"+ filename[:-4]+".json", 'w') as outfile:
            json.dump(annotations, outfile)

        #save aug image to a file
        imageio.imwrite(dataset.ROOT_IMG_DIR + set+"_"+aug_method+"//"+filename[:-4]+".tif", img_aug)