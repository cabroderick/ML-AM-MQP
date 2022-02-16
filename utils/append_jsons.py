'''
making a miserable script to fix miserable problems
there's probably so many better ways we could have implemented this project
but here we are
'''
import json
import os.path

IMG_DIRS = 'H6,H7,J0,J1,J3,J4,J5,J8,K0,K0R,K4,K5,Q6,Q8,R0'
ROOT_IMG_DIR = '../Stitched/'

for dir in IMG_DIRS.split(','):
    with open(ROOT_IMG_DIR + dir + '_merged.json') as f:
        merged_json = json.load(f)
    with open(ROOT_IMG_DIR + dir + '.json') as f:
        img_json = json.load(f)

    fields = 'version,flags,imagePath,imageData,imageHeight,imageWidth'
    for field in fields.split(','):
        merged_json[field] = img_json[field]

    with open(ROOT_IMG_DIR + dir + '_merged.json', 'w') as outfile:
        json.dump(merged_json, outfile)