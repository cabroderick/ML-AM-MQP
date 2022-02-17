import json
import os

dir = '../Data/Labels/Labeled K1/'

for path in os.listdir(dir):
    print(path)
    data = json.load(open(dir+path))
    new_path = path.replace('R', 'L')
    json.dump(data, open(dir+new_path, 'w'), indent=2)
    os.remove(dir+path)