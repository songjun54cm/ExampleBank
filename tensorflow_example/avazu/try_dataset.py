__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/18
import tensorflow as tf
import numpy as np

fields_name = ["id", "click", "hour", "C1", "banner_pos", "site_id", "site_domain", "site_category",
               "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
               "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
HOUR_MAP = {
    "00": "zz",
    "01": "zone",
    "02": "ztwo",
    "03": "zthr",
    "04": "zfour",
    "05": "zfive",
    "06": "zsix",
    "07": "zseven",
    "08": "zeight",
    "09": "znive",
    "10": "ten",
    "11": "elven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
    "21": "twentyone",
    "22": "twentytwo",
    "23": "twentythree"
}
hour_keys = [x[0] for x in HOUR_MAP.items()]
hour_vals = [x[1] for x in HOUR_MAP.items()]

hour_table = tf.contrib.lookup.HashTable(
    tf.contrib.lookup.KeyValueTensorInitializer(hour_keys, hour_vals), "default")

def parser_data(*value):
    fea = dict(zip(fields_name, value))
    fea["hour"] = tf.substr(fea["hour"], 6, 2)
    fea["hour"] = hour_table.lookup(fea["hour"])
    # fea["hour"] = tf.map_fn(lambda x: np.array(HOUR_MAP[x[0]]), fea["hour"])
    return fea

train_data_path = 'G://Datasets//avazuCTR//train.csv'
dataset = tf.contrib.data.CsvDataset(train_data_path, [[""]]*24, header=True, field_delim=",")


dataset = dataset.map(parser_data)
dataset = dataset.batch(2)
with tf.Session() as sess:
    iterator = dataset.make_initializable_iterator()
    sess.run(iterator.initializer)
    sess.run(tf.tables_initializer())
    next_element = iterator.get_next()
    for i in range(1):
        try:
            data_and_target = sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break
        print("\n\n", data_and_target)