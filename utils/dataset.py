import json
import os
import numpy as np
import cv2
from mrcnn import utils
from normalize_classname import normalize_classname

class Model_Dataset(utils.Dataset):

  # model constants, override as needed
  BASE_IMAGES_DIR = '/home/cabroderick/Data/Images/' # directory where all images can be found
  BASE_ANNOTATIONS_DIR = '/home/cabroderick/Data/Labels/' # directory where all images labels can be found
  # VALIDATION_PATH = 'validation.txt' # path to file containing validation images to skip during training
  IMAGES_DIRS = [] # list of image dirs to train on
  TRAIN_TEST_SPLIT = .8 # proportion of images to use for training set, remainder will be reserved for validation
  CLASSES = ['lack of fusion porosity', 'keyhole porosity', 'other'] # all annotation classes

  '''
  Loads the dataset
  validation: Indicates whether the current set is the validation set
  '''
  def load_dataset(self, validation=False):
    # f = open(self.VALIDATION_PATH, 'r+')
    # val_images = [line.strip() for line in f.readlines()]
    # f.close()

    image_paths = []
    annotation_paths = []
    image_ids = []

    for i in range(len(self.IMAGES_DIRS)):
      image_paths.append([])
      annotation_paths.append([])
      image_ids.append([])
      i_dir = self.BASE_IMAGES_DIR + self.IMAGES_DIRS[i] + '/'
      a_dir = self.BASE_ANNOTATIONS_DIR + 'Labeled ' + self.IMAGES_DIRS[i] + '/'
      for file in os.listdir(i_dir):
        i_id = file[:-4]
        # if i_id+'.tif' in val_images: # skip validation images
        #     continue
        image_ids[i].append(i_id)
        image_paths[i].append(i_dir+i_id+'.tif')
        annotation_paths[i].append(a_dir+i_id+'.json')

    if (len(image_paths) != len(annotation_paths)): # raise exception if mismatch betwaeen number of images and annotations
      raise(ValueError('Number of images and annotations must be equal'))

    # configure dataset
    for i in range(len(self.CLASSES)):
      self.add_class('dataset', i+1, self.CLASSES[i]) # add classes to model

    # add images and annotations to dataset, ensuring an even distribution
    for i in range(len(image_paths)):
      images = len(image_paths[i])
      train_images = int(images * self.TRAIN_TEST_SPLIT)
      val_images = int(images * (1 - self.TRAIN_TEST_SPLIT))
      if validation:
        for j in range(val_images):
          image_id = image_ids[i][j]
          image_path = image_paths[i][j]
          annotation_path = annotation_paths[i][j]

          mask, class_ids = self.extract_mask(image_path, annotation_path)

          if len(mask) != 0: # skip images with no annotations
              self.add_image('dataset',
                             image_id=image_id,
                             path=image_path,
                             mask=mask,
                             class_ids=class_ids)

      else:
        for j in range(train_images):
          image_id = image_ids[i][j + val_images]
          image_path = image_paths[i][j + val_images]
          annotation_path = annotation_paths[i][j + val_images]

          mask, class_ids = self.extract_mask(image_path, annotation_path)

          if len(mask) != 0:
              self.add_image('dataset',
                             image_id=image_id,
                             path=image_path,
                             mask=mask,
                             class_ids=class_ids)

  '''
  Extracts a mask from an image
  image_id: The image id to extract the mask from
  Returns a mask and a corresponding list of class ids
  '''
  def load_mask(self, image_id):

    info = self.image_info[image_id] # extract image info from data added earlier
    mask = info['mask']
    class_ids = info['class_ids']

    return mask, class_ids

  '''
  Extracts the mask data from an image and its respective annotation
  image_path: Path to the image
  annotation_path: Path to the annotation
  Returns a mask and a list of class ids
  '''
  def extract_mask(self, image_path, annotation_path):
    if not os.path.exists(annotation_path): # if the annotation path is not found, it is named differently than its source image
        annotation_path = annotation_path[:-5] + '_20X_YZ.json'

    # print(image_path, annotation_path)

    f_ann = open(annotation_path,)
    annotation_json = json.load(f_ann)

    if not annotation_json['shapes']: # if there are no annotations to be extracted
        return [], [] # empty list return values will be ignored and thus image is ignored

    class_ids = []
    image = cv2.imread(image_path)
    height = image.shape[0]
    width = image.shape[1]

    annotation_list = []
    [annotation_list.append(shape) for shape in annotation_json['shapes'] if shape['shape_type'] =='rectangle'
     and normalize_classname(shape['label']) != 'gas entrapment porosity'] # get annotations in a list
    mask = np.zeros([height, width, len(annotation_list)], dtype='uint8') # initialize array of masks for each bounding box

    for i in range(len(annotation_list)):
      a = annotation_list[i]

      # extract row and col data and crop image to annotation size
      col_min, col_max = int(min(a['points'][0][0], a['points'][1][0])), int(max(a['points'][0][0], a['points'][1][0]))
      row_min, row_max = int(min(a['points'][0][1], a['points'][1][1])), int(max(a['points'][0][1], a['points'][1][1]))
      col_min, col_max, row_min, row_max = self.normalize_dimensions(col_min, col_max, row_min, row_max)
      cropped_img = image[row_min:row_max, col_min:col_max]  # crop image to size of bounding box
      cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
      edged = cv2.Canny(cropped_img_gray, 30, 200)

      # apply contour to image and fill
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
      dilated = cv2.dilate(edged, kernel)
      contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      polygon = np.zeros(cropped_img.shape)
      color = [255, 255, 255]
      cv2.fillPoly(polygon, contours, color)

      # normalize polygon to all boolean values and insert into mask
      polygon_bool = np.alltrue(polygon == color, axis=2)
      mask[row_min:row_max, col_min:col_max, i] = polygon_bool

      # draw contour and mask
      # cv2.drawContours(edged, contours, -1, (0, 255, 0), 1)
      # imS = cv2.resize(edged, (512, 512))
      # cv2.imshow('Contours', imS)
      # cv2.waitKey(0)
      # cv2.imshow('Polygon', cv2.resize(polygon, (512, 512)))
      # cv2.waitKey(0)

      # extract class id and append to list
      class_label = normalize_classname(a['label'])
      class_id = self.CLASSES.index(class_label)
      class_ids.append(class_id)

    return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

  '''
  Ensures extracted row and column coords are not out of bounds
  '''
  def normalize_dimensions(self, col_min, col_max, row_min, row_max):
      return max(col_min, 0), col_max, max(row_min, 0), row_max