import tensorflow as tf

from image_data_pipeline import ImageDataPipeline

# DEFAULT GLOBAL PARAMETERS
# dimensions of dogs-vs-cats images
img_width = 150
img_height = 150

input_shape = (img_width, img_height, 3)

DEFAULT_TRAIN_LOG_DIR = "train_logs"
DEFAULT_SUMMARY_LOG_DIR = "summary_logs"

DEFAULT_TRAIN_DATA_DIR = "data/train"
DEFAULT_NB_TRAIN_SAMPLES = 2000
DEFAULT_VALIDATION_DATA_DIR = "data/validation"
DEFAULT_NB_VALIDATION_SAMPLES = 800

# Configurations
DEFAULT_BATCH_SIZE = 30
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.0005


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def build_cnn_model(x_input):
    # conv layer 1
    conv1 = tf.layers.conv2d(inputs=x_input, filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

    # conv layer 2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])

    # conv layer 3
    conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=[2, 2])

    # flattern
    shape = pool3.get_shape().as_list()
    # pprint.pprint(shape)

    pool3_flat = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

    dense = tf.layers.dense(inputs=pool3_flat, units=64, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.5)

    output = tf.layers.dense(inputs=dropout, units=2)

    return output


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
tf.app.flags.DEFINE_boolean("sync_flag", "false", "synchronized training")  # default to false instead of true
tf.app.flags.DEFINE_string("init_wait", "10", "worker initial wait for sync sessions")


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    task_index = int(FLAGS.task_index)
    init_wait = int(FLAGS.init_wait)

    synchronized_training = FLAGS.sync_flag

    tf.reset_default_graph()

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=task_index)

    if FLAGS.job_name == "ps":
        print("Parameter Server is started and ready ...")
        server.join()
    elif FLAGS.job_name == "worker":
        # To setup summary logs for chief worker
        is_chief = task_index == 0

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % int(FLAGS.task_index),
                cluster=cluster)):
            x_input = tf.placeholder("float", [None, img_width, img_height, 3])
            y_label = tf.placeholder("float", [None, 2])

            # we create a global_step tensor for distributed training
            # (a counter of iterations)
            global_step = tf.train.get_or_create_global_step()

            cnn_model = build_cnn_model(x_input)

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=cnn_model, labels=y_label)
            cost = tf.reduce_mean(loss)

            optimizer = tf.train.RMSPropOptimizer(learning_rate=DEFAULT_LEARNING_RATE, decay=0.9, epsilon=1e-8)
            train_op = optimizer.minimize(cost, global_step=global_step)

            init_op = tf.global_variables_initializer()

            epochs = DEFAULT_EPOCHS
            batch_size = DEFAULT_BATCH_SIZE

            n_batch = int(DEFAULT_NB_TRAIN_SAMPLES / batch_size)

            # [Between Graph Data Parallelism]
            # Prepare training data for each worker assuming each worker has its own local data directory to process
            image_data_pipeline = ImageDataPipeline()

            config = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d' % int(FLAGS.task_index)])

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   config=config,
                                                   is_chief=(int(FLAGS.task_index) == 0 and
                                                            (FLAGS.job_name == 'worker')),
                                                   checkpoint_dir="train_logs") as mon_sess:
                # if is_chief:
                #     print("Chief Worker initialize all variables ...")
                #     mon_sess.run(init_op)
                # else:
                #     time.sleep(10)

                validation_images, validation_labels = image_data_pipeline.get_validation_image_samples(
                                                                                (img_width, img_height),
                                                                                batch_size)

                for e in range(epochs):
                    print("At {}th epoch: ======================================================".format(e))

                    for b in range(n_batch):
                        images, labels = image_data_pipeline.get_next_train_image_batch(
                                                                    (img_width, img_height),
                                                                    batch_size)

                        mon_sess.run(train_op, feed_dict={x_input: images, y_label: labels})

                        if b > 0 and b % 32 == 0:
                            train_cost_val = mon_sess.run(cost, feed_dict={x_input: images, y_label: labels})
                            print("    At {}th step, the cost for training samples is {}".format(b, train_cost_val))

                    validation_cost_val = mon_sess.run(cost, feed_dict={x_input: validation_images,
                                                                        y_label: validation_labels})

                    print("    the cost for validation samples is {}\n".format(validation_cost_val))

            print("Training worker {} done!".format(task_index))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()