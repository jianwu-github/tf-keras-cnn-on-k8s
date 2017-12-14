import argparse

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard

from tf_keras_cnn_model_builder import KerasCnnModelBuilder

# dimensions of dogs-vs-cats images
img_width  = 150
img_height = 150

input_shape = (img_width, img_height, 3) if K.image_data_format() == 'channels_last' else (3, img_width, img_height)

CATS_DOGS_CATEGORY_LIST = ['cats', 'dogs']

DEFAULT_TRAIN_DATA_DIR = "data/train"
DEFAULT_NB_TRAIN_SAMPLES = 2000
DEFAULT_VALIDATION_DATA_DIR = "data/validation"
DEFAULT_NB_VALIDATION_SAMPLES = 800

DEFAULT_MODEL_DIR = "model"


def build_cnn_model(sample_shape):
    cnn_model_builder = KerasCnnModelBuilder(CATS_DOGS_CATEGORY_LIST)

    return cnn_model_builder.build_cnn_classification_model(sample_shape)


def train_cnn_model(train_data_dir, nb_train_samples,
                    validation_data_dir, nb_validation_samples,
                    batch_size, epochs):
    train_datagen = ImageDataGenerator(rescale=(1.0 / 255), shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=(1.0 / 255))

    print("Training Keras CNN Image Classifier: ========================")
    print("  Training Data Dir:   {}".format(train_data_dir))
    print("  Validation Data Dir: {}".format(validation_data_dir))
    print("  Categories/Labels:   {}".format(CATS_DOGS_CATEGORY_LIST))

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=CATS_DOGS_CATEGORY_LIST,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=CATS_DOGS_CATEGORY_LIST,
        class_mode='binary')

    train_steps = nb_train_samples // batch_size
    validation_steps = nb_validation_samples // batch_size

    tensorboard = TensorBoard(log_dir="summary_logs",
                              write_graph=True)

    cnn_model = build_cnn_model(input_shape)

    cnn_model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[tensorboard])

    model_file = DEFAULT_MODEL_DIR + "/cnn_model.h5"
    cnn_model.save(model_file)


parser = argparse.ArgumentParser(description="Training Keras CNN Dog-vs-Cat Image Classifier")
parser.add_argument('-ep', dest="epochs", type=int, default=10, help="Number of epochs")
parser.add_argument('-bs', dest="batch_size", type=int, default=16, help="Batch Size")

if __name__ == '__main__':
    parsed_args = parser.parse_args()

    epochs = parsed_args.epochs
    batch_size = parsed_args.batch_size
    train_data_dir = DEFAULT_TRAIN_DATA_DIR
    nb_train_samples = DEFAULT_NB_TRAIN_SAMPLES
    validation_data_dir = DEFAULT_VALIDATION_DATA_DIR
    nb_validation_samples = DEFAULT_NB_VALIDATION_SAMPLES

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.reset_default_graph()

    train_cnn_model(train_data_dir, nb_train_samples,
                    validation_data_dir, nb_validation_samples,
                    batch_size, epochs)

    # https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
