import cv2
from load_dataset_paths import load_dataset_paths
import random

OUT_DIR = '/home/cabroderick/test_set.txt'
ROOT_IMG_DIR = '/home/cabroderick/Data/Images/' # root directory where all JSONs are contained
ROOT_ANN_DIR = '/home/cabroderick/Data/Labels/'
SUB_DIRS = ['G0',
            'G8',
            'G9',
            'H0',
            'H4',
            'H5',
            'H6',
            'H7',
            'J3',
            'J4',
            'K0R',
            'K5',
            'Q0',
            'Q6',
            'R0',
            'R2',
            'R6'] # subdirs containing JSONs (remove 'Labeled' from dir name)

test_imgs = []
for dir in SUB_DIRS:
    img_dirs, ann_dirs = load_dataset_paths(ROOT_IMG_DIR, ROOT_ANN_DIR, [dir])
    imgs = []
    for path in img_dirs:
        id = path.split('/')[-1][:-4]
        imgs.append(id)
    test_imgs.append(random.choice(imgs) + '\n')

print(test_imgs)
f = open(OUT_DIR, 'a')
f.writelines(test_imgs)
f.close()