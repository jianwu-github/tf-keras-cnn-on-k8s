from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.models import Model

BINARY_CLASSIFICATION_MODEL = "cnn_binary_classification"
CATEGORICAL_CLASSIFICATION_MODEL = "cnn_categorical_classification"

DEFAULT_LOSS_FUNCTION = 'mean_squared_error'


class KerasCnnModelBuilder:

    def __init__(self, category_list=None):
        if category_list is None:
            self._category_list = []
            self._num_of_categories = 0
        else:
            self._category_list = category_list
            self._num_of_categories = len(category_list)

        if self._num_of_categories == 2:
            self._model_type = BINARY_CLASSIFICATION_MODEL
            self._loss_function = 'binary_crossentropy'
        elif self._num_of_categories > 2:
            self._model_type == CATEGORICAL_CLASSIFICATION_MODEL
            self._loss_function = 'categorical_crossentropy'
        else:
            self._model_type = None
            self._loss_function = 'mean_squared_error'

    def get_category_list(self):
        return self._category_list

    def build_cnn_classification_model(self, input_shape):
        """
        Building CNN Model using Keras 2 Functional API

        :param sample_shape:
        :return: keras cnn model
        """

        input_image = Input(shape=input_shape)

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

        cnn_classification_model = Model(inputs=input_image, outputs=prediction)

        cnn_classification_model.compile(loss=self._loss_function, optimizer='rmsprop', metrics=['accuracy'])

        return cnn_classification_model
