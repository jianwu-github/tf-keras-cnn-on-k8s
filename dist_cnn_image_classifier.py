import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras import backend as K

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_string("task_index", "0", "Index of task within the job")
tf.app.flags.DEFINE_string("log_path", "/tmp/train", "Log path")
tf.app.flags.DEFINE_string("data_dir", "/data", "Data dir path")
tf.app.flags.DEFINE_boolean("sync_flag", "true", "synchronized training")
tf.app.flags.DEFINE_string("init_wait", "10", "worker initial wait for sync sessions")

# dimensions of dogs-vs-cats images
img_width  = 150
img_height = 150

input_shape = (img_width, img_height, 3) if K.image_data_format() == 'channels_last' else (3, img_width, img_height)

DEFAULT_TRAIN_DATA_DIR = "data/train"
DEFAULT_NB_TRAIN_SAMPLES = 2000
DEFAULT_VALIDATION_DATA_DIR = "data/validation"
DEFAULT_NB_VALIDATION_SAMPLES = 800

DEFAULT_BATCH_SIZE = 15
DEFAULT_EPOCHS = 50


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


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    task_index = int(FLAGS.task_index)
    init_wait = int(FLAGS.init_wait)

    synchronized_training = FLAGS.sync_flag

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=task_index)

    if FLAGS.job_name == "ps":
        print("Parameter Server is started and ready ...")
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # set Keras learning phase to train
            K.set_learning_phase(1)
            # do not initialize variables on the fly
            K.manual_variable_initialization(True)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()