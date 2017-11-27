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
            self._train_dog_image_dir = DEFAULT_TRAIN_DATA_DIR + "/dogs"
            self._train_cat_image_dir = DEFAULT_TRAIN_DATA_DIR + "/cats"
            self._validation_dog_image_dir = DEFAULT_VALIDATION_DATA_DIR + "/dogs"
            self._validation_cat_image_dir = DEFAULT_VALIDATION_DATA_DIR + "/cats"
        else:
            self._train_dog_image_dir = data_dir + "/train/dogs"
            self._train_cat_image_dir = data_dir + "/train/cats"
            self._validation_dog_image_dir = data_dir + "/validation/dogs"
            self._validation_cat_image_dir = data_dir + "/validation/cats"

        self._train_dog_image_list = glob.glob(self._train_dog_image_dir + "/dog*.jpg")
        self._train_cat_image_list = glob.glob(self._train_cat_image_dir + "/cat*.jpg")

        self._num_train_dog_images = len(self._train_dog_image_list)
        self._num_train_cat_images = len(self._train_cat_image_list)

        # initialize the position at the end of list
        self._curr_train_dog_pos = self._num_train_dog_images - 1
        self._curr_train_cat_pos = self._num_train_cat_images - 1

        self._validation_dog_image_list = glob.glob(self._validation_dog_image_dir + "/dog*.jpg")
        self._validation_cat_image_list = glob.glob(self._validation_cat_image_dir + "/cat*.jpg")

        self._num_validation_dog_images = len(self._validation_dog_image_list)
        self._num_validation_cat_images = len(self._validation_cat_image_list)

    def _next_train_dog_pos(self):
        self._curr_train_dog_pos += 1
        if self._curr_train_dog_pos == self._num_train_dog_images:
            self._curr_train_dog_pos = 0

        return self._curr_train_dog_pos

    def _next_train_cat_pos(self):
        self._curr_train_cat_pos += 1
        if self._curr_train_cat_pos == self._num_train_cat_images:
            self._curr_train_cat_pos = 0

        return self._curr_train_cat_pos

    def resize_image(self, image, dim, interpolation):
        return cv2.resize(image, dim, interpolation=interpolation)

    def get_next_train_image_batch(self, image_dim, batch_size):
        images = []
        labels = []

        num_of_dogs = batch_size // 2
        num_of_cats = batch_size - num_of_dogs

        for d in range(num_of_dogs):
            dog_image_file = self._train_dog_image_list[self._next_train_dog_pos()]

            # convert color image to gray scale
            # dog_image = cv2.imread(dog_image_file, 0)
            dog_image = cv2.imread(dog_image_file)
            dog_image_shape = dog_image.shape

            interpolation = cv2.INTER_AREA if image_dim[0] > dog_image_shape[0] and image_dim[1] > dog_image_shape[1] else cv2.INTER_CUBIC

            images.append(self.resize_image(dog_image, image_dim, interpolation))
            labels.append(np.array(DEFAULT_DOG_ONE_HOT))

        for c in range(num_of_cats):
            cat_image_file = self._train_cat_image_list[self._next_train_cat_pos()]

            # convert color image to gray scale
            # cat_image = cv2.imread(cat_image_file, 0)
            cat_image = cv2.imread(cat_image_file)
            cat_image_shape = cat_image.shape

            interpolation = cv2.INTER_AREA if image_dim[0] > cat_image_shape[0] and image_dim[1] > cat_image_shape[1] else cv2.INTER_CUBIC

            images.append(self.resize_image(cat_image, image_dim, interpolation))
            labels.append(np.array(DEFAULT_CAT_ONE_HOT))

        # TODO: cache image data for speed?

        return images, labels

    def get_validation_image_samples(self, image_dim, sample_size):
        images = []
        labels = []

        num_of_dogs = sample_size // 2
        num_of_cats = sample_size - num_of_dogs

        dog_samples = np.random.choice(self._num_validation_dog_images, num_of_dogs)
        cat_samples = np.random.choice(self._num_validation_cat_images, num_of_cats)

        for d in dog_samples:
            dog_image_file = self._validation_dog_image_list[d]

            # convert color image to gray scale
            # dog_image = cv2.imread(dog_image_file, 0)
            dog_image = cv2.imread(dog_image_file)
            dog_image_shape = dog_image.shape

            interpolation = cv2.INTER_AREA if image_dim[0] > dog_image_shape[0] and image_dim[1] > dog_image_shape[1] else cv2.INTER_CUBIC

            images.append(self.resize_image(dog_image, image_dim, interpolation))
            labels.append(np.array(DEFAULT_DOG_ONE_HOT))

        for c in cat_samples:
            cat_image_file = self._validation_cat_image_list[c]

            # convert color image to gray scale
            # cat_image = cv2.imread(cat_image_file, 0)
            cat_image = cv2.imread(cat_image_file)
            cat_image_shape = cat_image.shape

            interpolation = cv2.INTER_AREA if image_dim[0] > cat_image_shape[0] and image_dim[1] > cat_image_shape[1] else cv2.INTER_CUBIC

            images.append(self.resize_image(cat_image, image_dim, interpolation))
            labels.append(np.array(DEFAULT_CAT_ONE_HOT))

        # TODO: cache image data for speed?

        return images, labels