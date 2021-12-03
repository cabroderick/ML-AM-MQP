'''
Visualization for weights obtained from training
Usage: python visualize.py [weights path] [image path]
'''

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import visualize
import sys
import cv2

if len(sys.argv) < 3:
    sys.exit('Insufficient arguments')

WEIGHTS_PATH = sys.argv[1]
IMG_PATH = sys.argv[2]
CLASSES = ['gas entrapment porosity', 'lack of fusion porosity', 'keyhole porosity']

############################################
# Configure model and load weights
############################################

class InferenceConfig(Config):
    NAME = 'inference'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3 + 1

model = MaskRCNN(mode='inference',
                 config=InferenceConfig(), model_dir='./')
model.load_weights(filepath=WEIGHTS_PATH, by_name=True)

#############################################
# Make prediction and visualize
#############################################

image = cv2.imread(IMG_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = model.detect([image], verbose=0)
r = r[0]

visualize.display_instances(image=image,
                                  boxes=r['rois'],
                                  masks=r['masks'],
                                  class_ids=r['class_ids'],
                                  class_names=CLASSES,
                                  scores=r['scores'])