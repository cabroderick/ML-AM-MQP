SCALE_FACTOR = .50
DIRS = ['H6']
BASE_IMAGE_DIR = '../Stitched images/'

import cv2

for dir in DIRS:
    img = cv2.imread(BASE_IMAGE_DIR + dir + '.png')
    scaled = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    cv2.imwrite(BASE_IMAGE_DIR + dir + '_50.png', scaled)