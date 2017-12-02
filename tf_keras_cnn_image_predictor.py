import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K

from tf_keras_cnn_model_builder import KerasCnnModelBuilder

# dimensions of dogs-vs-cats images
img_width  = 150
img_height = 150

input_shape = (img_width, img_height, 3) if K.image_data_format() == 'channels_last' else (3, img_width, img_height)

DEFAULT_TEST_DATA_DIR = "data/test"

DEFAULT_MODEL_DIR = "model"

DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR + "/cnn_model.h5"

CATS_DOGS_CATEGORY_LIST = ['cats', 'dogs']


def load_cnn_model(model_file):
    cnn_model_builder = KerasCnnModelBuilder(CATS_DOGS_CATEGORY_LIST)
    cnn_model = cnn_model_builder.build_cnn_classification_model(input_shape)
    cnn_model.load_weights(model_file)

    return cnn_model


def test_predictions():
    # prepare test data
    image_data_generator = ImageDataGenerator(rescale=(1.0 / 255))

    # load test dog images
    test_dog_image_list = glob.glob(DEFAULT_TEST_DATA_DIR + "/dogs/dog*.jpg")
    test_dog_images = []
    for dog_file in test_dog_image_list:
        dog_image = image.load_img(dog_file, target_size=(img_height, img_width))
        rescaled_dog_image = image_data_generator.standardize(image.img_to_array(dog_image))

        test_dog_images.append(rescaled_dog_image)

    # load test cat images
    test_cat_image_list = glob.glob(DEFAULT_TEST_DATA_DIR + "/cats/cat*.jpg")
    test_cat_images = []
    for cat_file in test_cat_image_list:
        cat_image = image.load_img(cat_file, target_size=(img_height, img_width))
        rescaled_cat_image = image_data_generator.standardize(image.img_to_array(cat_image))

        test_cat_images.append(rescaled_cat_image)

    # load trained cnn model
    cnn_model = load_cnn_model(DEFAULT_MODEL_FILE)

    print("\nTest Trained CNN Model with Dog Images: =======================")
    for dog_image in test_dog_images:
        prediction = cnn_model.predict(np.array([dog_image]), batch_size=1)
        print(" Predicted Probability of Dog Image is {}".format(prediction[0]))

    print("\nTest Trained CNN Model with Cat Images: =======================")
    for cat_image in test_cat_images:
        prediction = cnn_model.predict(np.array([cat_image]), batch_size=1)
        print(" Predicted Probability of Dog Image is {}".format(prediction[0]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()

    test_predictions()

    # https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
