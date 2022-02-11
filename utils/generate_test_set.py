import os
import random

OUT_DIR = '../test_set.txt'
ROOT_IMG_DIR = '../Data/Images/'
IMG_DIRS = ['H6', 'J8']

img_ids = [] # all image ids on a row basis for each set
current_row = 1

for i in range(len(IMG_DIRS)):
    dir = IMG_DIRS[i]
    img_ids.append([])
    current_ids = []
    for path in os.listdir(ROOT_IMG_DIR + dir + '/'):
        pos = path.split('_')[1].split('.')[0]
        row = int(pos[0])
        if row != current_row:
            img_ids[i].append(current_ids)
            current_ids = []
            current_row = row
        else:
            current_ids.append(path[:-4])
    img_ids[i].append(current_ids)

lines = [] # lines to write to output file
for set in img_ids:
    for subset in set:
       if subset:
           lines.append(random.choice(subset) + '\n')

f = open(OUT_DIR, 'w')
f.writelines(lines)
f.close()