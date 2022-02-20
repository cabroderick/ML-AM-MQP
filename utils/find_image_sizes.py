import cv2
from load_dataset_paths import load_dataset_paths

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

img_dirs, ann_dirs = load_dataset_paths(ROOT_IMG_DIR, ROOT_ANN_DIR, SUB_DIRS)

widths = []
heights = []

for dir in img_dirs:
    print(dir)
    img = cv2.imread(dir)
    width = img.shape[1]
    height = img.shape[0]
    widths.append(width)
    heights.append(height)

print('The minimum width is: ' + str(min(widths)))
print('The maximum width is: ' + str(max(widths)))
print('The minimum height is: ' + str(min(heights)))
print('The maximum height is: ' + str(max(heights)))