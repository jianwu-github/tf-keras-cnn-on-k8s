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

DEFAULT_TRAIN_DATA_DIR = "data/train"
DEFAULT_NB_TRAIN_SAMPLES = 2000
DEFAULT_VALIDATION_DATA_DIR = "data/validation"
DEFAULT_NB_VALIDATION_SAMPLES = 800


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

    cnn_model = build_cnn_model(input_shape)

    cnn_model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    cnn_model.save("model/cnn_model")


if __name__ == '__main__':
    train_data_dir = DEFAULT_TRAIN_DATA_DIR
    nb_train_samples = DEFAULT_NB_TRAIN_SAMPLES
    validation_data_dir = DEFAULT_VALIDATION_DATA_DIR
    nb_validation_samples = DEFAULT_NB_VALIDATION_SAMPLES
    epochs = 50
    batch_size = 16

    train_cnn_model(train_data_dir, nb_train_samples,
                    validation_data_dir, nb_validation_samples,
                    batch_size, epochs)


