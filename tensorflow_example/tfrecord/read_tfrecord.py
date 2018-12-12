__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/12/11
import argparse
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main_by_dataset(config):
    # a good reference:
    # https://github.com/YJango/TFRecord-Dataset-Estimator-API/blob/master/TensorFlow%20Dataset%20%2B%20TFRecords.ipynb
    data_path = "../data/tfrecord_test.tfrecords"
    dataset = tf.data.TFRecordDataset(data_path)
    def parse_function(serialized_example):
        fea_dics = {
            "label": tf.FixedLenFeature([], tf.int64),
            "fea1" : tf.FixedLenFeature([10], tf.int64),
            "fea2" : tf.FixedLenFeature([], tf.string)
        }
        parsed_example = tf.parse_single_example(serialized_example, fea_dics)
        parsed_example["fea2"] = tf.decode_raw(parsed_example["fea2"], tf.int64)
        return parsed_example

    new_dataset = dataset.map(parse_function)
    iterator = new_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        for batch_i in range(5):
            print("sample for iterator %d" % batch_i)
            label, fea1, fea2 = sess.run([next_element["label"],
                                         next_element["fea1"],
                                         next_element["fea2"]])
            print(label, fea1, fea2)

        shuffle_dataset = new_dataset.shuffle(buffer_size=10)
        iterator = shuffle_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        for batch_i in range(5):
            print("sample for shuffle %d" % batch_i)
            label, fea1, fea2 = sess.run([next_element["label"],
                                          next_element["fea1"],
                                          next_element["fea2"]])
            print(label, fea1, fea2)

        batch_dataset = shuffle_dataset.batch(4)
        iterator = batch_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        for batch_i in range(5):
            print("sample for batch %d" % batch_i)
            label, fea1, fea2 = sess.run([next_element["label"],
                                          next_element["fea1"],
                                          next_element["fea2"]])
            print(label, fea1, fea2)

        num_epochs = 2
        epoch_dataset = new_dataset.repeat(num_epochs)
        iterator = epoch_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        for batch_i in range(5):
            print("sample for epoch iter %d" % batch_i)
            label, fea1, fea2 = sess.run([next_element["label"],
                                          next_element["fea1"],
                                          next_element["fea2"]])
            print(label, fea1, fea2)


def main_by_queue(config):
    with tf.Session() as sess:
        data_path="../data/tfrecord_test.tfrecords"
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        feature={
            "label": tf.FixedLenFeature([], tf.int64),
            "fea1" : tf.FixedLenFeature([10], tf.int64),
            "fea2" : tf.FixedLenFeature([], tf.string)
        }
        features = tf.parse_single_example(serialized_example, features=feature)

        label = features["label"]
        fea1 = features["fea1"]

        fea2 = tf.decode_raw(features["fea2"], tf.int64)
        # shape must be clear , not implicit
        fea2 = tf.reshape(fea2, [5])

        lab, fe1, fe2 = tf.train.shuffle_batch([label, fea1, fea2], batch_size=2, capacity=10,
                                          num_threads=1, min_after_dequeue=5)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for batch_index in range(5):
            print("sample for batch %d" % batch_index)
            ll1, ff1, ff2 = sess.run([label, fea1, fea2])
            print(ll1, ff1, ff2)
            l1, f1, f2 = sess.run([lab, fe1, fe2])
            print(l1, f1, f2)

        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    # main_by_queue(config)
    main_by_dataset(config)