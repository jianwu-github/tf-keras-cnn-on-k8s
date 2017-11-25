import time

import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras import backend as K

# DEFAULT GLOBAL PARAMETERS
# dimensions of dogs-vs-cats images
img_width = 150
img_height = 150

input_shape = (img_width, img_height, 3) if K.image_data_format() == 'channels_last' else (3, img_width, img_height)

DEFAULT_TRAIN_LOG_DIR = "train_logs"
DEFAULT_SUMMARY_LOG_DIR = "summary_logs"

DEFAULT_TRAIN_DATA_DIR = "data/train"
DEFAULT_NB_TRAIN_SAMPLES = 2000
DEFAULT_VALIDATION_DATA_DIR = "data/validation"
DEFAULT_NB_VALIDATION_SAMPLES = 800

# Configurations
DEFAULT_BATCH_SIZE = 15
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.0005


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

    return cnn_model


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
# tf.app.flags.DEFINE_string("log_path", "train_logs", "Log path")
# tf.app.flags.DEFINE_string("data_dir", "data", "Data dir path")
tf.app.flags.DEFINE_boolean("sync_flag", "false", "synchronized training")  # default to false instead of true
tf.app.flags.DEFINE_string("init_wait", "10", "worker initial wait for sync sessions")


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
        # To setup summary logs for chief worker
        is_chief = task_index == 0

        # [Between Graph Data Parallelism]
        # Prepare training data for each worker assuming each worker has its own local data directory to process
        train_datagen = ImageDataGenerator(rescale=(1.0 / 255), shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        validation_datagen = ImageDataGenerator(rescale=(1.0 / 255))

        train_generator = train_datagen.flow_from_directory(
            DEFAULT_TRAIN_DATA_DIR,
            target_size=(img_width, img_height),
            batch_size=DEFAULT_BATCH_SIZE,
            class_mode='binary')

        validation_generator = validation_datagen.flow_from_directory(
            DEFAULT_VALIDATION_DATA_DIR,
            target_size=(img_width, img_height),
            batch_size=DEFAULT_BATCH_SIZE,
            class_mode='binary')

        train_steps = DEFAULT_NB_TRAIN_SAMPLES // DEFAULT_BATCH_SIZE
        validation_steps = DEFAULT_NB_VALIDATION_SAMPLES // DEFAULT_BATCH_SIZE

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % int(FLAGS.task_index),
                cluster=cluster)):
            # set Keras learning phase to train
            K.set_learning_phase(1)
            # do not initialize variables on the fly
            K.manual_variable_initialization(True)

            # Build Keras model
            model = build_cnn_model(input_shape)

            predictions = model.output
            targets = tf.placeholder(tf.float32, shape=None)

            # xent_loss = tf.reduce_mean(keras.losses.binary_crossentropy(targets, predictions))
            #
            # # apply regularizers if any
            # if model.regularizers:
            #     total_loss = xent_loss * 1.  # copy tensor
            #     for regularizer in model.regularizers:
            #         total_loss = regularizer(total_loss)
            # else:
            #     total_loss = xent_loss
            total_loss = tf.reduce_mean(keras.losses.binary_crossentropy(targets, predictions))

            # we create a global_step tensor for distributed training
            # (a counter of iterations)
            global_step = tf.train.get_or_create_global_step()

            # set up TF optimizer
            optimizer = tf.train.RMSPropOptimizer(learning_rate=DEFAULT_LEARNING_RATE, decay=0.9, epsilon=1e-8)

            # Set up model update ops (batch norm ops).
            # The gradients should only be computed after updating the moving average
            # of the batch normalization parameters, in order to prevent a data race
            # between the parameter updates and moving average computations.
            with tf.control_dependencies(model.updates):
                barrier = tf.no_op(name='update_barrier')

            # define gradient updates
            with tf.control_dependencies([barrier]):
                grads = optimizer.compute_gradients(
                    total_loss,
                    model.trainable_weights,
                    gate_gradients=tf.train.Optimizer.GATE_OP,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False)

                grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

            # define train tensor
            with tf.control_dependencies([grad_updates]):
                train_op = tf.identity(total_loss, name="train")

            # The StopAtStepHook handles stopping after running given steps.
            # hooks = [tf.train.StopAtStepHook(last_step=DEFAULT_TRAINING_STEPS)]
            hooks = None

            init_op = tf.global_variables_initializer()

            config = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d' % int(FLAGS.task_index)])

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   config=config,
                                                   is_chief=(int(FLAGS.task_index) == 0 and (FLAGS.job_name == 'worker')),
                                                   # is_chief=is_chief,
                                                   checkpoint_dir="train_logs",
                                                   hooks=hooks) as mon_sess:

                # if is_chief:
                #     print("Chief Worker initialize all variables ...")
                #     mon_sess.run(init_op)
                # else:
                #     time.sleep(10)

                for e in range(DEFAULT_EPOCHS):
                    print("run {} epoch".format(e))

                    # Training
                    curr_train_step = 0
                    for x_batch, y_batch in train_datagen.flow_from_directory(DEFAULT_TRAIN_DATA_DIR,
                                                                              target_size=(img_width, img_height),
                                                                              batch_size=DEFAULT_BATCH_SIZE,
                                                                              class_mode='binary'):

                        loss = mon_sess.run(train_op, feed_dict={model.inputs[0]: x_batch, targets: y_batch})

                        if curr_train_step % 40 == 0:
                            print("At {} epoch {} step, loss: {}".format(e, curr_train_step, loss))

                        curr_train_step += 1
                        if curr_train_step >= train_steps:
                            break

                    # # Using Validation Data to check
                    # validation_data, validation_labels = validation_datagen.flow_from_directory(DEFAULT_VALIDATION_DATA_DIR,
                    #                                                                             target_size=(img_width, img_height),
                    #                                                                             batch_size=DEFAULT_NB_VALIDATION_SAMPLES,
                    #                                                                             class_mode='binary')
                    #
                    # total_loss = mon_sess.run(total_loss, feed_dict={model.inputs[0]: validation_data, targets: validation_labels})
                    #
                    # print("At {} epoch, total loss on validation data set: {}".format(e, total_loss))

            print("Training worker {} done!".format(task_index))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
