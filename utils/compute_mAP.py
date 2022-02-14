'''
Computes mean average precision over a set of images
Usage: python compute_mAP.py [pre-trained weight path] [image dirs path] [test set path]
'''
import sys
import os
import mrcnn.model as modellib
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from mrcnn import utils
import numpy as np
from dataset import Model_Dataset

class InferenceConfig(Config):
    NAME = 'inference'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2 + 1

with open(sys.argv[2], 'r') as f:
    img_dirs = [line.rstrip('\n') for line in f.readlines()]
with open(sys.argv[3], 'r') as f:
    test_imgs = [line.rstrip('\n') for line in f.readlines()]

config = InferenceConfig()
dataset = Model_Dataset()
dataset.CLASSES = ['lack of fusion porosity', 'keyhole porosity']
dataset.ROOT_IMG_DIR = os.path.expanduser('~') + '/Data/Images/'
dataset.ROOT_ANNOTATION_DIR = os.path.expanduser('~') + '/Data/Labels/'
dataset.IMG_DIRS = img_dirs
dataset.TEST_SET = test_imgs
dataset.load_dataset_inference()
dataset.prepare()

model = MaskRCNN(mode='inference',
                 config=InferenceConfig(), model_dir='./')
model.load_weights(filepath=sys.argv[1], by_name=True)

APs = []
for image_id in dataset.image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))