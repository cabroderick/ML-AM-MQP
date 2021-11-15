# imports
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from cv2 import imread
import os
import json
import numpy as np
import urllib.request

# configure network

class CustomConfig(Config):
    NAME = "object"
    GPU_COUNT = 4
    IMAGES_PER_GPU = 2
    # number of classes (including background)
    NUM_CLASSES = 1 + 2
    STEPS_PER_EPOCH = 100
    LEARNING_RATE = .001

config = CustomConfig()

# set up dataset

class AMDataset(utils.Dataset):
  def load_dataset(self, validation=False):
    IMAGES_DIR = './ML-AM-MQP/Data/Trial/H6/'
    ANNOTATIONS_DIR = './ML-AM-MQP/Data/Trial/Labeled H6/'
    IMG_WIDTH = 1280
    IMG_HEIGHT = 1024

    self.add_class('dataset', 1, 'gas porosity')
    self.add_class('dataset', 2, 'lack of fusion porosity')

    val_images = 5 # keeps track of images to reserve for validation set
    total_images = len(os.listdir(IMAGES_DIR))
    for filename in os.listdir(IMAGES_DIR):
      if validation and val_images > 0:
        val_images -=1
        continue
      if (not validation) and val_images < total_images:
        val_images += 1
        continue

      image_id = filename[:-4]
      image_path = IMAGES_DIR + image_id + '.tif'
      annotation_path = ANNOTATIONS_DIR + image_id + '_labeled.json'
      self.add_image('dataset',
                     image_id=image_id, 
                     path=image_path, 
                     annotation=annotation_path,
                     width=IMG_WIDTH,
                     height=IMG_HEIGHT)

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
          col_s, col_e = int(box[key][0][0]), int(box[key][0][1])
          row_s, row_e = int(box[key][1][0]), int(box[key][1][1])
          masks[row_s:row_e, col_s:col_e, masks_index] = 1
          masks_index += 1
          class_ids.append(self.class_names.index(key))

    return masks, np.array(class_ids)

  def extract_boxes(self, filename): # helper to extract bounding boxes from json
      f = open(filename,)
      data = json.load(f)

      boxes = [] # store box coordinates in a dictionary corresponding to labels

      for i in data['shapes']:
        if i['shape_type'] == 'rectangle':
          box = {'gas porosity': [], 'lack of fusion porosity': []}
          label = i['label']
          box[label] = i['points']
          boxes.append(box)
 
      return boxes

# set up train and validation data

dataset_train = AMDataset()
dataset_train.load_dataset(validation=False)
dataset_train.prepare()

dataset_val = AMDataset()
dataset_val.load_dataset(validation=True)
dataset_val.prepare()

# configure model and load coco weights

model = MaskRCNN(mode='training', model_dir='./', config=CustomConfig())
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train model
model.train(train_dataset=dataset_train,
            val_dataset=dataset_val,
            learning_rate=.001,
            epochs=1,
            layers='heads')

# save training results to external file
model_path = 'custom_maskrcnn_weights.h5'
model.keras_model.save_weights(model_path)
