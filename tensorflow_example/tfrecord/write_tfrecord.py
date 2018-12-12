__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/12/11
import argparse
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def main(config):
    data_path = "../data/tfrecord_test.tfrecords"
    writer = tf.python_io.TFRecordWriter(data_path)
    for i in range(100):
        label = i
        fea1 = np.random.random_integers(0,100,size=(10))
        fea2 = np.random.random_integers(0,100, size=(5)).astype(np.int64).tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "fea2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[fea2])),
                "fea1": tf.train.Feature(int64_list=tf.train.Int64List(value=fea1))
            }
        ))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)