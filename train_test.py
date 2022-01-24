'''
Script to test training on high-dimensional images
'''

import os
import cv2
import numpy as np
from mrcnn import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

class CustomConfig(Config):
    NAME = "test_mcrnn"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    LEARNING_RATE = .001
    BATCH_SIZE = 1
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = [128, 128]

config = CustomConfig()

class CustomDataset(utils.Dataset):

  TRAIN_TEST_SPLIT = .8 # proportion of images to use for training set, remainder will be reserved for validation
  CLASSES = ['dummy class'] # all annotation classes
  IMAGE_DIR = 'Test_Images'

  def load_dataset(self):
      for i in range(len(self.CLASSES)):
          self.add_class('dataset', i + 1, self.CLASSES[i])

      for img in os.listdir('./' + self.IMAGE_DIR + '/'):
          image_path = './' + self.IMAGE_DIR + '/' + img
          print(image_path)
          self.add_image('dataset',
                         image_id=img,
                         path=image_path)

  def load_mask(self, image_id):
    image_path = self.image_info[image_id]['path']
    print(image_path)
    image = cv2.imread(image_path)
    height = image.shape[0]
    width = image.shape[1]

    mask = np.zeros([height, width, 1], dtype='uint8')
    return mask, [0]

dataset_train = CustomDataset()
dataset_train.load_dataset()
dataset_train.prepare()

dataset_val = CustomDataset()
dataset_val.load_dataset()
dataset_val.prepare()

# configure model
model = MaskRCNN(mode='training', model_dir='./test_model/', config=CustomConfig())

# train model
model.train(train_dataset=dataset_train,
           val_dataset=dataset_val,
           learning_rate=config.LEARNING_RATE,
           epochs=20,
           layers='heads')