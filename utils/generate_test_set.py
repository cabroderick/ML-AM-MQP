import os
import random

OUT_DIR = '/home/cabroderick/test_set.txt'
ROOT_IMG_DIR = '/home/cabroderick/Data/Images/'
IMG_DIRS = ['G0',
'G7',
'G8',
'G9',
'H0',
'H4',
'H5',
'H6',
'H7',
'H8',
'H9',
'J0',
'J1',
'J3',
'J4',
'J4R',
'J5',
'J7',
'J8',
'J9',
'K0',
'K0R',
'K1',
'K4',
'K5',
'Q0',
'Q3',
'Q4',
'Q5',
'Q6',
'Q9',
'R0',
'R2',
'R5',
'R6',
'R7'
]

img_ids = [] # all image ids on a row basis for each set
current_row = 1

for i in range(len(IMG_DIRS)):
    dir = IMG_DIRS[i]
    img_ids.append([])
    current_ids = []
    for path in os.listdir(ROOT_IMG_DIR + dir + '/'):
        if len(path.split('/')[-1]) < 6:
            current_ids.append(path[:-4])
        else:
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