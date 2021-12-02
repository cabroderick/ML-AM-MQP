'''
Mask RCNN model training on custom AM dataset for train use on WPI HPC cluster
Usage: python train.py [model name] [optional pre-trained weights file path]
'''

import json
import os
import sys
import numpy as np
import cv2
from mrcnn import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

if len(sys.argv) < 2: # ensure model name is included in arguments
  sys.exit('Insufficient arguments')

######################################
# Configuration
######################################
class CustomConfig(Config):
    NAME = "custom_mcrnn"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3 # 3 classes + background
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    LEARNING_RATE = .001
    BATCH_SIZE = 1

config = CustomConfig()
# config.display()

#######################################
# Dataset
#######################################
class CustomDataset(utils.Dataset):

  # define constants
  BASE_IMAGES_DIR = '../Data/Trial/' # directory where all images can be found
  BASE_ANNOTATIONS_DIR = '../Data/Trial/' # directory where all images labels can be found
  IMAGES_DIRS = ['H6/', 'H8/', 'J7/'] # list of directories where images are contained
  ANNOTATIONS_DIRS = ['Labeled H6/', 'Labeled H8/', 'Labeled J7/'] # corresponding list of directories where annotations are contained
  TRAIN_TEST_SPLIT = .8 # proportion of images to use for training set, remainder will be reserved for validation
  CLASSES = ['gas entrapment porosity', 'lack of fusion porosity', 'keyhole porosity'] # all annotation classes

  '''
  Loads the dataset
  validation: Indicates whether the current set is the validation set
  '''
  def load_dataset(self, validation=False):
    image_paths = []
    annotation_paths = []
    image_ids = []

    for i in range(len(self.IMAGES_DIRS)):
      i_dir = self.BASE_IMAGES_DIR + self.IMAGES_DIRS[i]
      a_dir = self.BASE_ANNOTATIONS_DIR + self.ANNOTATIONS_DIRS[i]
      for file in os.listdir(i_dir):
        i_id = file[:-4]
        image_ids.append(i_id)
        image_paths.append(i_dir+i_id+'.tif')
        annotation_paths.append(a_dir+i_id+'_20X_YZ.json')
    
    if (len(image_paths) != len(annotation_paths)): # raise exception if mismatch betwaeen number of images and annotations
      raise(ValueError('Number of images and annotations must be equal'))

    total_images = len(image_paths) # count of all images to be processed
    val_images = (int) (total_images * (1-self.TRAIN_TEST_SPLIT)) # the total number of images in the validation set

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

      image_id = image_ids[i]
      image_path = image_paths[i]
      annotation_path = annotation_paths[i]
      # print(image_id, image_path, annotation_path)

      mask, class_ids = self.extract_mask(image_path, annotation_path)

      self.add_image('dataset',
                     image_id=image_id, 
                     path=image_path,
                     mask=mask,
                     class_ids=class_ids)

  '''
  Extracts a mask from an image
  image_id: The image id to extract the mask from
  Returns a mask and a corresponding list of class ids
  '''
  def load_mask(self, image_id):
    info = self.image_info[image_id] # extract image info from data added earlier
    mask = info['mask']
    class_ids = info['class_ids']

    return mask, class_ids

  '''
  Extracts the mask data from an image and its respective annotation
  image_path: Path to the image
  annotation_path: Path to the annotation
  Returns a mask and a list of class ids
  '''
  def extract_mask(self, image_path, annotation_path):
    class_ids = []
    f_ann = open(annotation_path,)
    annotation_json = json.load(f_ann)
    image = cv2.imread(image_path)
    height = image.shape[0]
    width = image.shape[1]

    annotation_list = []
    [annotation_list.append(shape) for shape in annotation_json['shapes'] if shape['shape_type'] =='rectangle'] # get annotations in a list
    mask = np.zeros([height, width, len(annotation_list)], dtype='uint8') # initialize array of masks for each bounding box

    for i in range(len(annotation_list)):
      a = annotation_list[i]
      # extract row and col data and crop image to annotation size
      col_min, col_max = int(min(a['points'][0][0], a['points'][1][0])), int(max(a['points'][0][0], a['points'][1][0]))
      row_min, row_max = int(min(a['points'][0][1], a['points'][1][1])), int(max(a['points'][0][1], a['points'][1][1]))
      cropped_img = image[row_min:row_max, col_min:col_max]  # crop image to size of bounding box
      cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
      edged = cv2.Canny(cropped_img_gray, 30, 200)

      # apply contour to image and fill
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
      dilated = cv2.dilate(edged, kernel)
      contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      polygon = np.zeros(cropped_img.shape)
      color = [255, 255, 255]
      cv2.fillPoly(polygon, contours, color)

      # normalize polygon to all boolean values and insert into mask
      polygon_bool = np.alltrue(polygon == color, axis=2)
      mask[row_min:row_max, col_min:col_max, i] = polygon_bool

      # # draw contour and mask
      # cv2.drawContours(cropped_img, contours, -1, (0, 255, 0), 1)
      # imS = cv2.resize(cropped_img, (512, 512))
      # cv2.imshow('Contours', imS)
      # cv2.waitKey(0)
      # cv2.imshow('Polygon', cv2.resize(polygon, (512, 512)))
      # cv2.waitKey(0)

      # extract class id and append to list
      class_label = self.normalize_classname(a['label'])
      class_id = self.CLASSES.index(class_label)
      class_ids.append(class_id)

    return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

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

#######################################
# Training
#######################################

# set up train and validation data

dataset_train = CustomDataset()
dataset_train.load_dataset(validation=False)
dataset_train.prepare()

dataset_val = CustomDataset()
dataset_val.load_dataset(validation=True)
dataset_val.prepare()

# configure model

model = MaskRCNN(mode='training', model_dir='./'+sys.argv[1]+'/', config=CustomConfig())

if len(sys.argv) > 2: # optionally load pre-trained weights
  model.load_weights(sys.argv[2], by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# print summary
print(model.keras_model.summary())

# train model
model.train(train_dataset=dataset_train,
           val_dataset=dataset_val,
           learning_rate=config.LEARNING_RATE,
           epochs=1,
           layers='heads')

# save training results to external file
model.keras_model.save_weights(sys.argv[1]+'.h5')