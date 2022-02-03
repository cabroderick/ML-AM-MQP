'''
Generates random image of a specified size for testing
'''

import numpy
from PIL import Image

TOTAL_IMAGES = 10
IMAGE_SIZE = 10000
DESTINATION_DIR = 'Test_Images'

for i in range(TOTAL_IMAGES):
    imarray = numpy.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
    im.save('./' + DESTINATION_DIR + '/' + str(i) + '.png')