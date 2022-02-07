'''
Mask RCNN model training on custom AM dataset for train use on WPI HPC cluster
Usage: python train.py [model name] [pre-trained weights file path] [path to directory list] [path to test set list] [optional config file]
'''

import sys
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from utils.dataset import Model_Dataset

if len(sys.argv) < 5:
  sys.exit('Insufficient arguments')

######################################
# Configuration
######################################
class TrainConfig(Config):
    NAME = "mrcnn-model"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3 # 3 classes + background
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    LEARNING_RATE = .001
    BATCH_SIZE = 1

config = TrainConfig()

# load list of img dirs
paths_file = open(sys.argv[3], 'r')
img_dirs = paths_file.readlines()
[line.rstrip('\n') for line in img_dirs]

# load list of test images
paths_file_test = open(sys.argv[4], 'r')
test_imgs = paths_file_test.readlines()
[line.rstrip('\n') for line in test_imgs]

# load config file if specified
if len(sys.argv) > 5:
    config_file = open(sys.argv[5])
    config_args = config_file.readlines()
    [line.rstrip('\n') for line in config_args]
    for arg in config_args:
        arg, val = arg.replace(' ', '').split('=')
        if arg == 'LEARNING_RATE':
            TrainConfig.LEARNING_RATE = val
        elif arg == 'BATCH_SIZE':
            TrainConfig.BATCH_SIZE = val
        elif arg == 'STEPS_PER_EPOCH':
            TrainConfig.STEPS_PER_EPOCH = val
        elif arg == 'VALIDATION_STEPS':
            TrainConfig.VALIDATION_STEPS = val
        elif arg == 'IMAGE_MIN_DIM':
            TrainConfig.IMAGE_MIN_DIM = val
        elif arg == 'IMAGE_MAX_DIM':
            TrainConfig.IMAGE_MAX_DIM = val
        elif arg == 'IMAGES_PER_GPU':
            TrainConfig.IMAGES_PER_GPU = val

#######################################
# Training
#######################################

# set up train and validation data
dataset_train = Model_Dataset()
dataset_train.IMG_DIRS = img_dirs
dataset_train.TEST_SET = test_imgs
dataset_train.load_dataset(validation=False)
dataset_train.prepare()

dataset_val = Model_Dataset()
dataset_val.IMG_DIRS = img_dirs
dataset_val.TEST_SET = test_imgs
dataset_val.load_dataset(validation=True)
dataset_val.prepare()

# configure model
model = MaskRCNN(mode='training', model_dir='./'+sys.argv[1]+'/', config=TrainConfig())

model.load_weights(sys.argv[3], by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train model
model.train(train_dataset=dataset_train,
           val_dataset=dataset_val,
           learning_rate=config.LEARNING_RATE,
           epochs=40,
           layers='heads')

# save training results to external file
model.keras_model.save_weights(sys.argv[1]+'.h5')