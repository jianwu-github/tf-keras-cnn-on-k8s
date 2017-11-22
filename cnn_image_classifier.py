import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras import backend as K

# dimensions of dogs-vs-cats images
img_width  = 150
img_height = 150

input_shape = (img_width, img_height, 3) if K.image_data_format() == 'channels_last' else (3, img_width, img_height)


def build_cnn_model(sample_shape):
    cnn_model = Sequential(name="cnn_model")

    cnn_model.add(Conv2D(32, (3, 3), input_shape=sample_shape))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(Conv2D(32, (3, 3)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(Conv2D(64, (3, 3)))
    cnn_model.add(Activation('relu'))
    cnn_model(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(64))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Dense(1))
    cnn_model.add(Activation('sigmoid'))

    cnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return cnn_model
