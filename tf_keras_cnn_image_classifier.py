import argparse
import gc

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard

# dimensions of dogs-vs-cats images
img_width  = 150
img_height = 150

input_shape = (img_width, img_height, 3) if K.image_data_format() == 'channels_last' else (3, img_width, img_height)

DEFAULT_TRAIN_DATA_DIR = "data/train"
DEFAULT_NB_TRAIN_SAMPLES = 2000
DEFAULT_VALIDATION_DATA_DIR = "data/validation"
DEFAULT_NB_VALIDATION_SAMPLES = 800


def build_cnn_model(sample_shape):
    """
    Building CNN Model using Keras 2 Functional API

    :param sample_shape:
    :return: keras cnn model
    """

    input_image = Input(shape=sample_shape)

    conv_1 = Conv2D(32, (3, 3), activation='relu')(input_image)
    pool_1 = MaxPooling2D((2, 2))(conv_1)

    conv_2 = Conv2D(32, (3, 3), activation='relu')(pool_1)
    pool_2 = MaxPooling2D((2, 2))(conv_2)

    conv_3 = Conv2D(64, (3, 3), activation='relu')(pool_2)
    pool_3 = MaxPooling2D((2, 2))(conv_3)

    flatten = Flatten()(pool_3)
    dense = Dense(64, activation='relu')(flatten)
    dropout = Dropout(0.5)(dense)

    prediction = Dense(1, activation='sigmoid')(dropout)

    cnn_model = Model(inputs=input_image, outputs=prediction)

    cnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return cnn_model


def train_cnn_model(train_data_dir, nb_train_samples,
                    validation_data_dir, nb_validation_samples,
                    batch_size, epochs):
    train_datagen = ImageDataGenerator(rescale=(1.0 / 255), shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=(1.0 / 255))

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    train_steps = nb_train_samples // batch_size
    validation_steps = nb_validation_samples // batch_size

    tensorboard = TensorBoard(log_dir="summary_logs")

    cnn_model = build_cnn_model(input_shape)

    cnn_model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[tensorboard])

    cnn_model.save("model/cnn_model.h5")


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
