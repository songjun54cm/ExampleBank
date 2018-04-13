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

    save_folder = 'output'
    saver_dir = os.path.join(save_folder, config['mode'], 'saver')
    saver_folder = os.path.join(saver_dir, config['mode'], 'saver_epoch_100')

    sess = tf.Session()

    meta_path = os.path.join(saver_folder, 'model.meta')
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess, tf.train.latest_checkpoint(saver_folder))

    graph = tf.get_default_graph()
    label_input_layer = graph.get_tensor_by_name("input_label:0")
    fea_input_layer = graph.get_tensor_by_name("input_feature:0")
    pred_accuracy = graph.get_tensor_by_name("pred_accuracy:0")

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

