__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/12/12
import argparse
import argparse
import numpy as np
import tensorflow as tf
import random
import os
import time


def load_data(epoch_num, batch_size):
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

    train_dataset = tf.data.Dataset.from_tensor_slices({
        "fea": train_data,
        "label": train_labels
    })\
        .repeat(epoch_num) \
        .batch(batch_size) \
        .shuffle(buffer_size=10*batch_size) \
        .prefetch(buffer_size=10*batch_size)
    train_iter = train_dataset.make_one_shot_iterator()
    train_next_elmt = train_iter.get_next()

    test_dataset = tf.data.Dataset.from_tensor_slices({
        "fea": eval_data,
        "label": eval_labels
    })\
        .batch(batch_size) \
        .prefetch(buffer_size=10*batch_size)
    test_iter = test_dataset.make_initializable_iterator()
    test_next_elmt = test_iter.get_next()

    return train_next_elmt, test_next_elmt


def main(config):
    input_label = tf.placeholder(tf.int64, [None], name="input_label")
    input_fea = tf.placeholder(tf.float32, [None,28,28,1], name="input_fea")
    conv1 = tf.layers.conv2d(
        inputs=input_fea,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    logits = tf.layers.dense(inputs=dropout, units=10)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')
    pred_class = tf.argmax(input=logits, axis=1, name='pred_class')
    pred_accuracy = tf.reduce_mean(tf.to_float(tf.equal(pred_class, input_label)), name='pred_accuracy')
    loss = tf.losses.sparse_softmax_cross_entropy(labels=input_label, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss)

    epoch_num = 100
    batch_size = 1000
    train_next_elmt, test_next_elmt = load_data(epoch_num, batch_size)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        step = 0
        try:
            while True:
                start_time = time.time()
                fea, label = sess.run([train_next_elmt["fea"],
                                       train_next_elmt["label"]])
                _, loss_val = sess.run([train_op, loss],
                                       feed_dict={
                                           input_fea: fea,
                                           input_label: label
                                       })
                if step % 1000 == 0:
                    duration = time.time() - start_time
                    print("step %d: loss = %f (%.3f sec)" % (step, loss_val, duration))
                if step % 10000 == 0:
                    accs = []
                    try:
                        start_time = time.time()
                        while True:
                            fea, label = sess.run([test_next_elmt["fea"],
                                                   test_next_elmt["label"]])
                            acc = sess.run(pred_accuracy,
                                           feed_dict={
                                               input_fea: fea,
                                               input_label: label
                                           })
                            accs.append(acc)
                    except tf.errors.OutOfRangeError:
                        acc = np.average(accs)
                        duration = time.time() - start_time
                        print("average acc: %f (%.3f sec)" % (acc, duration))
                        print("Done testing.")
        except tf.errors.OutOfRangeError:
            print("Done training for %d epochs, %d steps." % (epoch_num, step))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)