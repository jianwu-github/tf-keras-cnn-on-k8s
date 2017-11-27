import glob

import cv2
import numpy as np


DEFAULT_TRAIN_DATA_DIR = "data/train"
DEFAULT_NB_TRAIN_SAMPLES = 2000
DEFAULT_VALIDATION_DATA_DIR = "data/validation"
DEFAULT_NB_VALIDATION_SAMPLES = 800

DEFAULT_DOG_LABEL = 1
DEFAULT_CAT_LABEL = 0

DEFAULT_DOG_ONE_HOT = [1, 0]
DEFAULT_CAT_ONE_HOT = [0, 1]


class ImageDataPipeline:
    def __init__(self, data_dir=None):
        if data_dir is None:
            self._dog_image_dir = DEFAULT_TRAIN_DATA_DIR + "/dogs"
            self._cat_image_dir = DEFAULT_TRAIN_DATA_DIR + "/cats"
        else:
            self._dog_image_dir = data_dir + "/train/dogs"
            self._cat_image_dir = data_dir + "/train/cats"

        self._dog_image_list = glob.glob(self._dog_image_dir + "/dog*.jpg")
        self._cat_image_list = glob.glob(self._cat_image_dir + "/cat*.jpg")

        self._num_dog_images = len(self._dog_image_list)
        self._num_cat_images = len(self._cat_image_list)

        # initialize the position at the end of list
        self._curr_dog_pos = self._num_dog_images - 1
        self._curr_cat_pos = self._num_cat_images - 1

    def _next_dog_pos(self):
        self._curr_dog_pos += 1
        if self._curr_dog_pos == self._num_dog_images:
            self._curr_dog_pos = 0

        return self._curr_dog_pos

    def _next_cat_pos(self):
        self._curr_cat_pos += 1
        if self._curr_cat_pos == self._num_cat_images:
            self._curr_cat_pos = 0

        return self._curr_cat_pos

    def resize_image(self, image, dim, interpolation):
        return cv2.resize(image, dim, interpolation=interpolation)

    def get_next_image_batch(self, image_dim, batch_size):
        images = []
        labels = []

        num_of_dogs = batch_size // 2
        num_of_cats = batch_size - num_of_dogs

        for d in range(num_of_dogs):
            dog_image_file = self._dog_image_list[self._next_dog_pos()]

            # convert color image to gray scale
            dog_image = cv2.imread(dog_image_file, 0)
            dog_image_shape = dog_image.shape

            interpolation = cv2.INTER_AREA if image_dim[0] > dog_image_shape[0] and image_dim[1] > dog_image_shape[1] else cv2.INTER_CUBIC

            images.append(self.resize_image(dog_image, image_dim, interpolation))
            labels.append(np.array(DEFAULT_DOG_ONE_HOT))

        for c in range(num_of_cats):
            cat_image_file = self._dog_image_list[self._next_cat_pos()]

            # convert color image to gray scale
            cat_image = cv2.imread(cat_image_file, 0)
            cat_image_shape = cat_image.shape

            interpolation = cv2.INTER_AREA if image_dim[0] > cat_image_shape[0] and image_dim[1] > cat_image_shape[1] else cv2.INTER_CUBIC

            images.append(self.resize_image(cat_image, image_dim, interpolation))
            labels.append(np.array(DEFAULT_CAT_ONE_HOT))

        # TODO: cache image data for speed?

        return images, labels
