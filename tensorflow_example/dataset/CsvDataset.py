__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/18
import tensorflow as tf
import numpy as np

"""
file1.csv contains following content:
ID  nr1 nr2 nr3 nr4 nr5 next_nr
1   1   2   3   4   5   6
2   2   3   4   5   6   7
3   3   4   5   6   7   8
4   4   5   6   7   8   9
5   5   6   7   8   9   10
6   6   7   8   9   10  11
7   7   8   9   10  11  12
8   8   9   10  11  12  13
9   9   10  11  12  13  14
10  10  11  12  13  14  15
"""


ITERATOR_BATCH_SIZE = 2
NR_EPOCHS = 3

train1_path = "file1.csv"

dataset = tf.contrib.data.CsvDataset(train1_path,
                                     [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                                     header=True, field_delim=' ')

dataset = dataset.map(lambda *x: tf.convert_to_tensor(x))
dataset = dataset.batch(ITERATOR_BATCH_SIZE)

with tf.Session() as sess:
    for i in range (NR_EPOCHS):
        print('\nepoch: ', i)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        while True:
            try:
                data_and_target = sess.run(next_element)
            except tf.errors.OutOfRangeError:
                break
            print("\n\n", data_and_target)