from PIL import Image, ImageDraw
import os
import json

# list of directories where images are contained
IMAGES_DIRS = ['G0/', 'G9/', 'H0/', 'H4/', 'H5/', 'H6/', 'H8/', 'H9/', 'J0/', 'J1/', 'J3/', 'J4/', 'J7/',
               'J8/', 'K0/', 'Q0/', 'Q3/', 'Q5/', 'Q9/', 'R2/', 'R6/','R7/']
IMAGE_ROOT = "./Images/"
LABELS_ROOT = "./Labels/"
X = 1280
Y = 1024

color_dict = {'gas entrapment porosity': "blue", 'lack of fusion porosity': "red", 'keyhole porosity': "purple",
              "other": "green"}


def normalize_classname(class_name):  # normalize the class name to one used by the model
    class_name = class_name.lower()  # remove capitalization
    class_name = class_name.strip()  # remove leading and trailing whitespace
    classes_dict = {  # dictionary containing all class names used in labels and their appropriate model class name
        'gas entrapment porosity': 'gas entrapment porosity',
        'keyhole porosity': 'keyhole porosity',
        'lack of fusion porosity': 'lack of fusion porosity',
        'fusion porosity': 'lack of fusion porosity',
        'gas porosity': 'gas entrapment porosity',
        'lack-of-fusion': 'lack of fusion porosity',
        'keyhole': 'keyhole porosity',
        'other': 'other',
        'lack of fusion': 'lack of fusion porosity'
    }
    return classes_dict.get(class_name)

def get_max_bb(points):
    x, y = zip(*points)
    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)
    return [(min_x, min_y), (max_x, max_y)]

def overlap_area(bb1, bb2):
    BUFFER = 2 #pixels
    #Add buffer border around to make sure overlap area positive when edges touch
    x1, y1, w1, h1 = min(bb1[0][0], bb1[1][0])-BUFFER,min(bb1[0][1], bb1[1][1])-BUFFER, max(bb1[0][0], bb1[1][0])-min(
        bb1[0][0],bb1[1][0])+BUFFER,max(bb1[0][1], bb1[1][1])-min(bb1[0][1], bb1[1][1])+BUFFER
    x2, y2, w2, h2 = min(bb2[0][0], bb2[1][0])-BUFFER,min(bb2[0][1], bb2[1][1])-BUFFER, max(bb2[0][0], bb2[1][0])-max(bb2[0][0],
                    bb2[1][0])+BUFFER, max(bb2[0][1], bb2[1][1])-min(bb2[0][1], bb2[1][1])+BUFFER
    x = max(x1, x2)
    y = max(y1, y2)
    w = min(x1 + w1, x2 + w2) - x
    h = min(y1 + h1, y2 + h2) - y
    w = w if w > 0 else 0
    h = h if h > 0 else 0
    return w*h

def stitch_all():
    for set in IMAGES_DIRS:
        annotations = {}
        annotations['shapes'] = []

        all_files = os.listdir(IMAGE_ROOT + set)
        x_y = [(int(x.split("_")[1].split(".")[0][0]),int(x.split("_")[1].split(".")[0][1:])) for x in all_files]
        x_dim = max(x_y)[1]
        y_dim = max(x_y)[0]

        image = Image.new('RGB', (X * x_dim, Y * y_dim))

        for filename in os.listdir(IMAGE_ROOT + set):
            img_path = IMAGE_ROOT + set + filename
            coordinates = filename.split("_")
            row = int(coordinates[1].split(".")[0][0])
            col = int(coordinates[1].split(".")[0][1:])

            im = Image.open(img_path)
            x_shift = (col - 1) * X
            y_shift = (row - 1) * Y
            image.paste(im, (x_shift, y_shift))

            labels_file_path = LABELS_ROOT + "Labeled " + set + filename[:-4] + ".json"
            if os.path.exists(labels_file_path):
                f_ann = open(labels_file_path, )
            else:
                labels_file_path = LABELS_ROOT + "Labeled " + set + filename[:-4] + "_20X_YZ.json"
                f_ann = open(labels_file_path, )
            annotation_json = json.load(f_ann)

            for label in annotation_json["shapes"]:
                new_label = {}
                new_label["group_id"] = None
                new_label["flags"] = []
                new_label["shape_type"] = label["shape_type"]
                new_label['label'] = normalize_classname(label['label'])

                shape = label["points"]
                new_points = [[x_shift + p[0], y_shift + p[1]] for p in shape]
                new_label["points"] = new_points

                annotations["shapes"].append(new_label)
        image2 = image.copy()
        image.save("Stitched/" + set[:-1] + ".png")

        img_annotated = ImageDraw.Draw(image)

        for defect in annotations["shapes"]:
            if defect["shape_type"] == "rectangle":
                xy = [(defect["points"][0][0], defect["points"][0][1]), (defect["points"][1][0], defect["points"][1][1])]
                img_annotated.rectangle(xy, fill=None, outline=color_dict[defect["label"]], width=5)
            if defect["shape_type"] == "polygon":
                xy = [tuple(x) for x in defect["points"]]
                img_annotated.polygon(xy, fill=None, outline=color_dict[defect["label"]], width=5)
        image.save("Stitched/" + set[:-1] + "_annotated.png")

        # annotations_list = [(instance['label'], instance['points']) for instance in annotations['shapes']]
        # merged_annotations = []
        # merged_list = []
        # for idx, bb in enumerate(annotations_list):
        #     if idx in merged_list:
        #         continue
        #     merged = False
        #     for idx2, bb2 in enumerate(annotations_list, idx+1):
        #         if bb[0] == bb2[0]: #if the label is the same
        #             if overlap_area(bb[1], bb2[1]) > 0:
        #                 #they overlap
        #                 p1 = [bb[1][0], bb[1][1], [bb[1][0][0],bb[1][1][1]], [bb[1][1][0],bb[1][0][1]]]
        #                 p2 = [bb2[1][0], bb2[1][1], [bb2[1][0][0],bb2[1][1][1]], [bb2[1][1][0],bb2[1][0][1]]]
        #                 merged_annotations.append((bb[0], get_max_bb(p1+p2)))
        #                 print((bb[0], get_max_bb(p1+p2)))
        #                 merged = True
        #                 merged_list.append(idx2)
        #
        #         if merged:
        #             break
        #     if not merged:
        #         merged_annotations.append(bb)
        #
        # img_annotated2 = ImageDraw.Draw(image2)
        # for defect in merged_annotations:
        #     xy = [(defect[1][0][0], defect[1][0][1]), (defect[1][1][0], defect[1][1][1])]
        #     img_annotated2.rectangle(xy, fill=None, outline=color_dict[defect[0]], width=5)
        # image2.save("Stitched/" + set[:-1] + "_annotated_merged.png")


# merge_how=[["12", "13"], ["14", "24"], ["17", "18", "27", "28"]]
#
# def stitch_set(set, merge_way_raw, folder_name):
#     naming_scheme = os.listdir(IMAGE_ROOT + set)[0].split("_")
#     merge_way_processed = []
#     for k in merge_way_raw:
#         new_k = [(int(v[0]), int(v[1:])) for v in k]
#         merge_way_processed.append(new_k)
#     for merged_set in merge_way_processed:
#         dim_x = max(merged_set)[0]-min(merged_set)[0]+1
#         dim_y = max(merged_set)[1]-min(merged_set)[1]+1
#         image = Image.new('RGB', (X * dim_x, Y * dim_y))
#
#         for im in merged_set:
#             print(im)
#             #get path
#             path = naming_scheme[0] + "_"+str(im[0])+str(im[1])+"_"+"_".join(naming_scheme[2:])
#             #paste image where it belongs in the merged image
#             im = Image.open(IMAGE_ROOT + set+path)
#             x_shift = (im[0]-dim_x - 1) * X
#             y_shift = (im[1]-dim_y - 1) * Y
#             image.paste(im, (x_shift, y_shift))
#
#
#
#
# stitch_set("G0/", merge_how, "G0 merge test")



