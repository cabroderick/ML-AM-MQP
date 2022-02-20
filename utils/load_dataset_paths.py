import os

def load_dataset_paths(root_img_dir, root_annotation_dir, dirs):
    image_paths = []
    annotation_paths = []

    for i in range(len(dirs)):
        i_dir = root_img_dir + dirs[i] + '/'
        a_dir = root_annotation_dir + 'Labeled ' + dirs[i] + '/'
        for file in os.listdir(i_dir):
            i_id = file[:-4]
            if os.path.exists(i_dir + i_id + '.tif'):
                image_paths.append(i_dir + i_id + '.tif')
            else:
                image_paths.append(i_dir + i_id + '.png')
            annotation_paths.append(a_dir + i_id + '.json')

    if len(image_paths) == len(annotation_paths):
        return image_paths, annotation_paths
    else:
        return None, None
