"""
Author: songjun
Date: 2018/4/13
Description:
Usage:
"""
import argparse
import tensorflow as tf
import numpy as np
import os


def main(config):
    print('loading data.')
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    print('train data shape %s' % str(train_data.shape))
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    print('train label shape %s' % str(train_labels.shape))
    eval_data = mnist.test.images  # Returns np.array
    eval_data = np.reshape(eval_data, (eval_data.shape[0], 28, 28, 1))
    print('eval data shape %s' % str(eval_data.shape))
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    print('eval labels shape %s' % str(eval_labels.shape))


    # Input Layer
    label_input_layer = tf.placeholder(tf.int64, [None], name='input_label')
    fea_input_layer = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_feature')

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=fea_input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    logits = tf.layers.dense(inputs=dropout, units=10)

    probabilities = tf.nn.softmax(logits, name='softmax_tensor')
    pred_class = tf.argmax(input=logits, axis=1, name='pred_class')
    pred_accuracy = tf.reduce_mean(tf.to_float(tf.equal(pred_class, label_input_layer)), name='pred_accuracy')
    # pred_accuracy = tf.metrics.accuracy(labels=label_input_layer, predictions=pred_class, name='pred_accuracy')

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=label_input_layer, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

    save_folder = 'output'
    saver_dir = os.path.join(save_folder, config['mode'], 'saver')
    saver_folder = os.path.join(saver_dir, config['mode'], 'saver_epoch_100')

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(saver_folder))

    acc = sess.run(pred_accuracy, feed_dict={fea_input_layer: eval_data, label_input_layer: eval_labels})
    print('evaluate accuracy: %f' % acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    if config['mode'] == 'gpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    elif config['mode'] == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main(config)

