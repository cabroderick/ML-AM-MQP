'''
Mask RCNN model training on custom AM dataset for train use on WPI HPC cluster
Usage: python train.py [model name] [optional pre-trained weights file path]
'''

import sys
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from utils.dataset import Model_Dataset

if len(sys.argv) < 2: # ensure model name is included in arguments
  sys.exit('Insufficient arguments')

######################################
# Configuration
######################################
class TrainConfig(Config):
    NAME = "custom_mcrnn"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3 # 3 classes + background
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    LEARNING_RATE = .001
    BATCH_SIZE = 1

config = TrainConfig()

#######################################
# Training
#######################################

# set up train and validation data

dataset_train = Model_Dataset()
dataset_train.load_dataset(validation=False)
dataset_train.prepare()

dataset_val = Model_Dataset()
dataset_val.load_dataset(validation=True)
dataset_val.prepare()

# configure model
model = MaskRCNN(mode='training', model_dir='./'+sys.argv[1]+'/', config=TrainConfig())

if len(sys.argv) > 2: # optionally load pre-trained weights
  model.load_weights(sys.argv[2], by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train model
model.train(train_dataset=dataset_train,
           val_dataset=dataset_val,
           learning_rate=config.LEARNING_RATE,
           epochs=40,
           layers='heads')

# save training results to external file
model.keras_model.save_weights(sys.argv[1]+'.h5')