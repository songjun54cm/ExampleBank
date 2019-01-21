__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/18
import tensorflow as tf
import numpy as np

fields_name = ["id", "click", "hour", "C1", "banner_pos", "site_id", "site_domain", "site_category",
               "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
               "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
def parser_data(*value):
    fea = dict(zip(fields_name, value))
    fea["hour"] = tf.substr(fea["hour"], 6, 2)
    # fea["hour"] = tf.map_fn(lambda x: x[-2:], fea["hour"])
    return fea

train_data_path = 'G://Datasets//avazuCTR//train.csv'
dataset = tf.contrib.data.CsvDataset(train_data_path, [[""]]*24, header=True, field_delim=",")


dataset = dataset.map(parser_data)
dataset = dataset.batch(2)
with tf.Session() as sess:
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    for i in range(1):
        try:
            data_and_target = sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break
        print("\n\n", data_and_target)