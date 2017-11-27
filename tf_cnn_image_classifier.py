import argparse
import pprint

import numpy as np
import tensorflow as tf

from image_data_pipeline import ImageDataPipeline

# dimensions of dogs-vs-cats images
img_width  = 150
img_height = 150

input_shape = (img_width, img_height, 3)

DEFAULT_TRAIN_DATA_DIR = "data/train"
DEFAULT_NB_TRAIN_SAMPLES = 2000
DEFAULT_VALIDATION_DATA_DIR = "data/validation"
DEFAULT_NB_VALIDATION_SAMPLES = 800

DEFAULT_LEARNING_RATE = 0.0005

MAX_FLOAT_VAL = np.finfo(np.float32).max

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def build_cnn_model(x_input):
    conv1 = tf.layers.conv2d(inputs=x_input, filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

    conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])

    conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=[2, 2])

    #flattern
    shape = pool3.get_shape().as_list()
    pprint.pprint(shape)

    pool3_flat = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

    dense = tf.layers.dense(inputs=pool3_flat, units=64, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.5)

    output = tf.layers.dense(inputs=dropout, units=2)

    return output


def train_cnn_model(epochs, batch_size):
    tf.reset_default_graph()

    x_input = tf.placeholder("float", [None, img_width, img_height, 3])
    y_label = tf.placeholder("float", [None, 2])

    global_step = tf.train.get_or_create_global_step()

    cnn_model = build_cnn_model(x_input)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=cnn_model, labels=y_label)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=DEFAULT_LEARNING_RATE, decay=0.9, epsilon=1e-8)
    train_op = optimizer.minimize(cost, global_step=global_step)

    init_op = tf.global_variables_initializer()

    image_data_pipeline = ImageDataPipeline()

    n_batch = int(DEFAULT_NB_TRAIN_SAMPLES / batch_size)

    with tf.Session() as sess:
        sess.run(init_op)

        validation_images, validation_labels = image_data_pipeline.get_validation_image_samples((img_width, img_height), batch_size)

        for e in range(epochs):
            print("At {}th epoch: ======================================================".format(e))
            for b in range(n_batch):
                images, labels = image_data_pipeline.get_next_train_image_batch((img_width, img_height), batch_size)

                sess.run(train_op, feed_dict={x_input: images, y_label: labels})

                if b > 0 and b % 32 == 0:
                    train_cost_val = sess.run(cost, feed_dict={x_input: images, y_label: labels})
                    print("    At {}th step, the cost for training samples is {}".format(b, train_cost_val))

            validation_cost_val = sess.run(cost, feed_dict={x_input: validation_images, y_label: validation_labels})
            print("    the cost for validation samples is {}\n".format(validation_cost_val))


parser = argparse.ArgumentParser(description="Training TensorFlow CNN Dog-vs-Cat Image Classifier")
parser.add_argument('-ep', dest="epochs", type=int, default=10, help="Number of epochs")
parser.add_argument('-bs', dest="batch_size", type=int, default=16, help="Batch Size")

if __name__ == '__main__':
    parsed_args = parser.parse_args()

    epochs = parsed_args.epochs
    batch_size = parsed_args.batch_size

    tf.logging.set_verbosity(tf.logging.DEBUG)

    train_cnn_model(epochs, batch_size)