'''
Simple script to ensure every image has a matching annotation - all images without annotations are removed
'''

import os

ROOT_IMG_DIR = '../Data/Images/' # directory where all images can be found
ROOT_ANNOTATION_DIR = '../Data/Labels/' # directory where all images labels can be found
IMG_DIRS = ['G9']

for i in range(len(IMG_DIRS)):
      i_dir = ROOT_IMG_DIR + IMG_DIRS[i] + '/'
      a_dir = ROOT_ANNOTATION_DIR + 'Labeled ' + IMG_DIRS[i] + '/'
      for file in os.listdir(i_dir):
        i_id = file[:-4]
        image_path = i_dir+i_id+'.tif'
        annotation_path = a_dir+i_id+'.json'
        # print(image_path, annotation_path)
        if not os.path.exists(annotation_path):
            print(image_path, annotation_path)
            print('MISSING ANNOTATION')
            os.remove(image_path)