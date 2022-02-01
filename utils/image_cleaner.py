'''
Removes 20X_YZ at the end of image names
'''

import cv2
import os

ROOT_DIR = '../Data/Images/' # root directory where all JSONs are contained
SUB_DIRS = ['R6'] # subdirs containing JSONs (remove 'Labeled' from dir name)

for dir in SUB_DIRS:
    dir_path = ROOT_DIR + dir + '/'
    for file in os.listdir(dir_path):
        img_path = ROOT_DIR + dir + '/' + file
        print(img_path)
        if img_path[-11:] == '_20X_YZ.tif':
            img = cv2.imread(img_path)
            os.remove(img_path)
            img_path = img_path[:-11] + img_path[-4:]
            cv2.imwrite(img_path, img)


