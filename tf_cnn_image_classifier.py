import argparse

import numpy as np
import tensorflow as tf

from image_data_pipeline import ImageDataPipeline

# dimensions of dogs-vs-cats images
img_width  = 150
img_height = 150

input_shape = (img_width, img_height, 1)

DEFAULT_TRAIN_DATA_DIR = "data/train"
DEFAULT_NB_TRAIN_SAMPLES = 2000
DEFAULT_VALIDATION_DATA_DIR = "data/validation"
DEFAULT_NB_VALIDATION_SAMPLES = 800

DEFAULT_LEARNING_RATE = 0.0005

MAX_FLOAT_VAL = np.finfo(np.float32).max

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def build_cnn_model(x_input):
    #conv1
    w = init_weights([3, 3, 1, 32])  # 3x3x1 conv, 32 outputs
    conv1 = tf.nn.conv2d(x_input, w, strides=[1, 1, 1, 1], padding='VALID')
    conv1_a = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #conv2
    w2 = init_weights([1, 74, 74, 32])  # 3x3x1 conv, 32 outputs
    conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='VALID')
    conv2_a = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # conv3
    w3 = init_weights([3, 3, 1, 64])  # 3x3x1 conv, 64 outputs
    conv3 = tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding='VALID')
    conv3_a = tf.nn.relu(conv3)
    pool3 = tf.nn.max_pool(conv3_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # flattern
    shape = pool3.get_shape().as_list()
    flat = tf.reshape(pool3, [shape[0], shape[1] * shape[2] * shape[3]])

    # output
    w4 = init_weights([shape[1] * shape[2] * shape[3], 2])

    output = tf.matmul(flat, w4)

    return output


def train_cnn_model(epochs, batch_size):
    tf.reset_default_graph()

    x_input = tf.placeholder("float", [None, img_width, img_height, 1])
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

        for e in range(epochs):

            min_cost = MAX_FLOAT_VAL
            for b in range(n_batch):
                images, labels = image_data_pipeline.get_next_image_batch((img_width, img_height), batch_size)

                cost = sess.run(train_op, feed_dict={x_input: images, y_label: labels})

                if min_cost < cost:
                    min_cost = cost

            print("Min Cost for {} epoch: {}".format(e, min_cost))


parser = argparse.ArgumentParser(description="Training TensorFlow CNN Dog-vs-Cat Image Classifier")
parser.add_argument('-ep', dest="epochs", type=int, default=10, help="Number of epochs")
parser.add_argument('-bs', dest="batch_size", type=int, default=16, help="Batch Size")

if __name__ == '__main__':
    parsed_args = parser.parse_args()

    epochs = parsed_args.epochs
    batch_size = parsed_args.batch_size

    tf.logging.set_verbosity(tf.logging.DEBUG)

    train_cnn_model(epochs, batch_size)