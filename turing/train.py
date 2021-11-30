'''
Mask RCNN model training on custom AM dataset for train use on WPI HPC cluster
Usage: python train.py [model name] [optional pre-trained weights file path]
'''

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import utils
from mrcnn.model import log
from cv2 import imread
import os
import json
import numpy as np
import sys
import skimage.draw
import matplotlib.pyplot as plt

if len(sys.argv) < 2: # ensure model name is included in arguments
  sys.exit('Insufficient arguments')

######################################
# Configuration
######################################
class CustomConfig(Config):
    NAME = "custom_mcrnn"
    # GPU_COUNT = 1
    # IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3 # 3 classes + background
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    LEARNING_RATE = .001
    # specify image size for resizing
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512
    BATCH_SIZE = 1

config = CustomConfig()
# config.display()

#######################################
# Dataset
#######################################
class CustomDataset(utils.Dataset):

  # define constants
  BASE_IMAGES_DIR = os.path.expanduser('~') + '/ML-AM-MQP/Data/Trial/' # directory where all images can be found
  BASE_ANNOTATIONS_DIR = os.path.expanduser('~') + '/ML-AM-MQP/Data/Trial/' # directory where all images labels can be found
  IMAGES_DIRS = ['H6/', 'H8/', 'J7/'] # list of directories where images are contained
  ANNOTATIONS_DIRS = ['Labeled H6/', 'Labeled H8/', 'Labeled J7/'] # corresponding list of directories where annotations are contained
  TRAIN_TEST_SPLIT = .8 # proportion of images to use for training set, remainder will be reserved for validation
  CLASSES = ['gas entrapment porosity', 'lack of fusion porosity', 'keyhole porosity'] # all annotation classes
  IMG_WIDTH = 1280
  IMG_HEIGHT = 1024
  # SCALE = -1 # to be used for resize_mask
  # PADDING = -1 # to be used for resize_mask

  '''
  Loads the dataset
  validation: Indicates whether the current set is the validation set
  '''
  def load_dataset(self, validation=False):
    image_paths = []
    annotation_paths = []

    for subdir in self.IMAGES_DIRS:
      [image_paths.append(self.BASE_IMAGES_DIR + subdir + dir) for dir in sorted(os.listdir(self.BASE_IMAGES_DIR+subdir))] # create the list of all image paths
    for subdir in self.ANNOTATIONS_DIRS:
      [annotation_paths.append(self.BASE_ANNOTATIONS_DIR + subdir + dir) for dir in sorted(os.listdir(self.BASE_ANNOTATIONS_DIR+subdir))] # create the list of all annotation paths
    
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

      image_path = image_paths[i]
      annotation_path = annotation_paths[i]
      image_id = image_path.split('/')[-1][:-4] # split the string by the '/' delimiter, get last element (filename), and remove file extension

      polygons, num_ids = self.extract_polygons(image_path, annotation_path)

      self.add_image('dataset',
                     image_id=image_id, 
                     path=image_path, 
                     annotation=annotation_path,
                     width=self.IMG_WIDTH,
                     height=self.IMG_HEIGHT,
                     polygons=polygons,
                     num_ids=num_ids)

  '''
  Extracts a mask from an image
  image_id: The image id to extract the mask from
  Returns a mask and a corresponding list of class ids
  '''
  def load_mask(self, image_id): # extracts all masks corresponding to a single image id
    info = self.image_info[image_id] # extract image info from data added earlier
    mask = np.zeros([info['height'], info['width'], len(info['polygons'])], dtype='uint8') # initialize array of masks for each bounding box
    num_ids = info['num_ids']

    for i, p in enumerate(info['polygons']):
      rr, cc = skimage.draw.polygon(p['y'], p['x'])
      mask[rr, cc, i] = 1

    return mask.astype(np.bool), np.array(num_ids, dtype=np.int32)

  '''
  Extracts the polygon data from an image and its respective annotation
  image_path: Path to the image
  annotation_path: Path to the annotation
  Returns a list of polygons and a list of class ids
  '''
  def extract_polygons(self, image_path, annotation_path): # helper to extract bounding boxes from json
    boxes = []
    num_ids = []
    f_ann = open(annotation_path,)
    annotation = json.load(f_ann)
    image = imread(image_path)

    # extract coordinate data (only from rectangles for now)
    for shape in annotation['shapes']:
      if shape['shape_type'] == 'rectangle':
        col_min, col_max = int(shape['points'][0][0]), int(shape['points'][1][0])
        row_min, row_max = int(shape['points'][0][1]), int(shape['points'][1][1])
        print(col_min, col_max, row_min, row_max)
        class_label = self.normalize_classname(shape['label'])
        num_id = self.CLASSES.index(class_label) # add id corresponding to label to ids list
        num_ids.append(num_id) # append to ids list
        cropped_img = image[col_min:col_max, row_min:row_max] # crop image to size
        imgplot = plt.imshow(image)
        plt.show()

    return boxes, num_ids

  # def load_image(self, image_id): # override load image to enable resizing
  #    # Load image
  #   image = skimage.io.imread(self.image_info[image_id]['path'])
  #   # If grayscale. Convert to RGB for consistency.
  #   if image.ndim != 3:
  #       image = skimage.color.gray2rgb(image)
  #   # If has an alpha channel, remove it for consistency
  #   if image.shape[-1] == 4:
  #       image = image[..., :3]

  #   image, window, scale, padding, _ = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM, max_dim=config.IMAGE_MAX_DIM, mode='square') # resize to dims specified by config
  #   self.SCALE = scale
  #   self.PADDING = padding
  #   return image

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

exit(0)

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