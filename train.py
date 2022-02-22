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
    NAME = "mrcnn_model"
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 4 + 1 # 4 classes + background
    STEPS_PER_EPOCH = 100 # 798 dataset size / batch size of 2 (2 image/GPU * 1 GPU)
    VALIDATION_STEPS = 5
    LEARNING_RATE = .0001 # optimized from hyperparameter tuning
    BACKBONE = "resnet50"

config = TrainConfig()
config.display()

# load list of img dirs
paths_file = open(sys.argv[3], 'r')
img_dirs = paths_file.readlines()
img_dirs = [line.rstrip('\n') for line in img_dirs]
print('Training on sets: ' + str(img_dirs))

# load list of test images
paths_file_test = open(sys.argv[4], 'r')
test_imgs = paths_file_test.readlines()
test_imgs = [line.rstrip('\n') for line in test_imgs]

# load config file if specified
if len(sys.argv) > 5:
    config_file = open(sys.argv[5])
    config_args = config_file.readlines()
    config_args = [line.rstrip('\n') for line in config_args]
    for arg in config_args:
        arg, val = arg.replace(' ', '').split('=')
        if arg == 'LEARNING_RATE':
            TrainConfig.LEARNING_RATE = float(val)
        elif arg == 'STEPS_PER_EPOCH':
            TrainConfig.STEPS_PER_EPOCH = int(val)
        elif arg == 'VALIDATION_STEPS':
            TrainConfig.VALIDATION_STEPS = int(val)
        elif arg == 'LEARNING_MOMENTUM':
            TrainConfig.LEARNING_MOMENTUM = float(val)
        print(arg + ' set to ' + val)

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

model.load_weights(sys.argv[2], by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train model
model.train(train_dataset=dataset_train,
           val_dataset=dataset_val,
           learning_rate=config.LEARNING_RATE,
           epochs=250,
           layers='heads')

# save training results to external file
model.keras_model.save_weights('./'+sys.argv[1]+'/'+sys.argv[1]+'.h5')