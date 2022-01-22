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
