# Mask RCNN model training on custom AM dataset for train use on WPI HPC cluster
# Usage: python train.py [model name] [optional pre-trained weights file path]

# imports
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from cv2 import imread
import os
import json
import numpy as np
import urllib.request
import sys

if not sys.argv: # ensure model name is included in arguments
  sys.exit('Insufficient arguments')

# configure network
class CustomConfig(Config):
    NAME = "custom_mcrnn"
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3 # 3 classes + background
    STEPS_PER_EPOCH = 100
    LEARNING_RATE = .001

config = CustomConfig()

# set up dataset
class AMDataset(utils.Dataset):
  # define constants
  BASE_IMAGES_DIR = os.path.expanduser('~') + '/ML-AM-MQP/Data/Trial/' # directory where all images can be found
  BASE_ANNOTATIONS_DIR = os.path.expanduser('~') + '/ML-AM-MQP/Data/Trial/' # directory where all images labels can be found
  IMAGES_DIRS = ['H6/', 'H8/', 'J7/'] # list of directories where images are contained
  ANNOTATIONS_DIRS = ['Labeled H6/', 'Labeled H8/', 'Labeled J7/'] # corresponding list of directories where annotations are contained

  TRAIN_TEST_SPLIT = .8 # proportion of images to use for training set, remainder will be reserved for validation
  CLASSES = ['gas entrapment porosity', 'lack of fusion porosity', 'keyhole porosity'] # all annotation classes

  IMG_WIDTH = 1280
  IMG_HEIGHT = 1024

  def load_dataset(self, validation=False):

    image_paths = [] # list of all paths to images to be processed
    annotation_paths = [] # list of all paths to annotations to be processed

    for subdir in self.IMAGES_DIRS:
      [image_paths.append(self.BASE_IMAGES_DIR + subdir + dir) for dir in sorted(os.listdir(self.BASE_IMAGES_DIR+subdir))] # create the list of all image paths
    for subdir in self.ANNOTATIONS_DIRS:
      [annotation_paths.append(self.BASE_ANNOTATIONS_DIR + subdir + dir) for dir in sorted(os.listdir(self.BASE_ANNOTATIONS_DIR+subdir))] # create the list of all annotation paths
    
    if (len(image_paths) != len(annotation_paths)): # raise exception if mismatch betwaeen number of images and annotations
      raise(ValueError('Number of images and annotations must be equal'))

    total_images = len(image_paths) # count of all images to be processed
    val_images = (int) (total_images * (1-self.TRAIN_TEST_SPLIT)) # the total number of images in the validation set

    print('Total images: ' + str(total_images))
    print('Validation images: ' + str(val_images))
    print('Training images: ' + str(total_images - val_images))

    # configure dataset
    for i in range(len(self.CLASSES)):
      self.add_class('dataset', i+1, self.CLASSES[i]) # add classes to model

    val_images_counter = val_images # counter to keep track of remaining images for validation set

    for i in range(total_images):
      if validation and val_images_counter > 0:
        val_images_counter -=1
        continue
      if (not validation) and val_images_counter < total_images:
        val_images_counter += 1
        continue

      image_path = image_paths[i]
      annotation_path = annotation_paths[i]
      print("Image path: " + image_path)
      print("Annotation path: " + annotation_path)
      image_id = image_path.split('/')[-1][:-4] # split the string by the '/' delimiter, get last element (filename), and remove file extension
      print("Image id: " + image_id)

      self.add_image('dataset',
                     image_id=image_id, 
                     path=image_path, 
                     annotation=annotation_path,
                     width=self.IMG_WIDTH,
                     height=self.IMG_HEIGHT)

  def load_mask(self, image_id):
    class_ids = list() # list of class ids corresponding to each mask in the mask list
    image_info = self.image_info[image_id] # extract image info from data added earlier

    width = image_info['width']
    height = image_info['height']
    path = image_info['annotation']

    masks_index = 0 # keep track of index for use in masks

    boxes = self.extract_boxes(path) # extract mask data from json file
    masks = np.zeros([height, width, len(boxes)], dtype='uint8') # initialize array of masks for each bounding box
    for i in range(len(boxes)):
      box = boxes[i]
      for key in box:
        if (box[key]): # make sure box is not empty
          col_s, col_e = int(box[key][0][0]), int(box[key][1][0])
          row_s, row_e = int(box[key][0][1]), int(box[key][1][1])
          print("Columns: " + str(col_s) + ", " + str(col_e))
          print("Rows: " + str(row_s) + ", " + str(row_e))
          masks[row_s:row_e, col_s:col_e, masks_index] = 1
          masks_index += 1
          class_ids.append(self.class_names.index(key))

    return masks, np.array(class_ids)

  def extract_boxes(self, filename): # helper to extract bounding boxes from json
      f = open(filename,)
      data = json.load(f)

      boxes = [] # store box coordinates in a dictionary corresponding to labels

      # extract coordinate data (only from rectangles for now)
      for rect in data['shapes']:
        if rect['shape_type'] == 'rectangle':
          box = {} # dictionary that contains a class and its corresponding list of points
          for c in self.CLASSES: # initialize dictionary keys
            box[c] = []
          label = self.normalize_classname(rect['label']) # get the label name from the JSON and fix name if needed
          print(rect)
          print(label)
          box[label] = rect['points'] # set the key value of the dictionary to the points extracted
          boxes.append(box) # add to list of extracted boxes
 
      return boxes

  def normalize_classname(self, class_name): # normalize the class name to one used by the model
    class_name = class_name.lower() # remove capitalization
    classes_dict = { # dictionary containing all class names used in labels and their appropriate model class name
      'gas entrapment porosity' : 'gas entrapment porosity',
      'keyhole porosity' : 'keyhole porosity',
      'lack of fusion porosity' : 'lack of fusion porosity',
      'fusion porosity' : 'lack of fusion porosity',
      'gas porosity' : 'gas entrapment porosity'
    }
    return classes_dict.get(class_name)

# set up train and validation data

dataset_train = AMDataset()
dataset_train.load_dataset(validation=False)
dataset_train.prepare()

dataset_val = AMDataset()
dataset_val.load_dataset(validation=True)
dataset_val.prepare()
dataset_val.load_mask(12)

# train model

model_coco = MaskRCNN(mode='training', model_dir='./'+sys.argv[0]+'/', config=CustomConfig())

if len(sys.argv) > 1: # optionally load pre-trained weights
  model_coco.load_weights(sys.argv[1], by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train model
model_coco.train(train_dataset=dataset_train,
           val_dataset=dataset_val,
           learning_rate=.001,
           epochs=5,
           layers='heads')

# save training results to external file
model_coco.keras_model.save_weights(sys.argv[0]+'.h5')