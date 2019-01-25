__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/18
import argparse
import tensorflow as tf
import pickle
import os
import numpy as np
from settings import DATA_DIR
import json

Fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
          'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
          'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
          'device_conn_type','click']

fields_dict = {}
for field in Fields:
    with open(os.path.join(DATA_DIR, "field2formField", "%s.pkl"%field), "rb") as f:
        fields_dict[field] = pickle.load(f)

# Field_Value_Map = {}
# for fname in Fields:
#     items = fields_dict[fname].items()
#     keys = [x[0] for x in items]
#     vals = [x[1] for x in items]
#     Field_Value_Map[fname] = tf.contrib.lookup.HashTable(
#         tf.contrib.lookup.KeyValueTensorInitializer(keys, vals), "other")


def get_columns(fields_dict):
    def get_categorial_column(field_name):
        col = tf.feature_column.categorical_column_with_vocabulary_list(field_name, list(set(fields_dict[field_name].values())))
        return col
    column_dict = {
        # "hour": tf.feature_column.categorial_column_with_identity("hour", 24),
        "hour": get_categorial_column("hour"),
        "C1": get_categorial_column("C1"),
        "C14": get_categorial_column("C14"),
        "C15": get_categorial_column("C15"),
        "C16": get_categorial_column("C16"),
        "C17": get_categorial_column("C17"),
        "C18": get_categorial_column("C18"),
        "C19": get_categorial_column("C19"),
        "C20": get_categorial_column("C20"),
        "C21": get_categorial_column("C21"),
        "banner_pos": get_categorial_column("banner_pos"),
        "site_id": get_categorial_column("site_id"),
        "site_domain": get_categorial_column("site_domain"),
        "site_category": get_categorial_column("site_category"),
        "app_domain": get_categorial_column("app_domain"),
        "app_id": get_categorial_column("app_id"),
        "app_category": get_categorial_column("app_category"),
        "device_model": get_categorial_column("device_model"),
        "device_type": get_categorial_column("device_type"),
        "device_id": get_categorial_column("device_id"),
        "device_conn_type": get_categorial_column("device_conn_type")
    }

    column_names = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                    'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                    'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
                    'device_conn_type']

    # wide columns
    wide_columns = [column_dict[cname] for cname in column_names]
    for i in range(len(column_names)):
        for j in range(i+1, len(column_names)):
            cross_column = tf.feature_column.crossed_column([column_names[i], column_names[j]], hash_bucket_size=1000)
            wide_columns.append(cross_column)

    deep_columns = [tf.feature_column.indicator_column(column_dict[cname]) for cname in column_names]
    # deep_columns = [tf.feature_column.embedding_column(column_dict[cname], dimmension=10) for cname in column_names]

    return wide_columns, deep_columns


def train_in_fun(data_path):
    fields_name = ["id", "click", "hour", "C1", "banner_pos", "site_id", "site_domain", "site_category",
                   "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
                   "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
    def parse_csv(*value):
        fea = dict(zip(fields_name, value))
        # fea["hour"] = tf.substr(fea["hour"], 6, 2)
        label = tf.equal(fea["click"], "1")
        # for k in Fields:
        #     fea[k] = Field_Value_Map[k].lookup(fea[k])
        return fea, label

    record_defaults = [[""]] * len(fields_name)
    dataset = tf.contrib.data.CsvDataset(data_path, record_defaults, header=True)
    dataset = dataset.map(parse_csv, num_parallel_calls=1). \
        batch(512).shuffle(buffer_size=10000).repeat(1)
    return dataset


def train_input_fn():
    train_data_path = 'G://Datasets//avazuCTR//form_train.csv'
    return train_in_fun(train_data_path)


def valid_input_fn():
    valid_data_path = 'G://Datasets//avazuCTR//form_train_valid.csv'
    return train_in_fun(valid_data_path)


def the_test_input_fn():
    field_names = ["id", "hour", "C1", "banner_pos", "site_id", "site_domain", "site_category",
                   "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
                   "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
    def parse_test_csv(*value):
        fea = dict(zip(field_names, value))
        # fea["hour"] = tf.substr(fea["hour"], 6, 2)
        return fea
    train_data_path = 'G://Datasets//avazuCTR//form_test.csv'
    record_defaults = [[""]] * len(field_names)
    dataset = tf.contrib.data.CsvDataset(train_data_path, record_defaults, header=True)
    dataset = dataset.map(parse_test_csv, num_parallel_calls=1). \
        batch(512).shuffle(buffer_size=10000).repeat(NUM_EPOCH)
    return dataset


def main():
    wide_column, deep_column = get_columns(fields_dict)
    hidden_units = [200, 200, 100]

    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(
                                      inter_op_parallelism_threads=2,
                                      intra_op_parallelism_threads=2)
    )
    model = tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns=wide_column,
        dnn_feature_columns=deep_column,
        dnn_hidden_units=hidden_units,
        config=run_config
    )

    train_epoch = 3
    for n in range(train_epoch):
        model.train(input_fn=train_input_fn)
        results = model.evaluate(input_fn=valid_input_fn)
        for key in sorted(results):
            print("%s: %s", (key, results[key]))



if __name__ == "__main__":
    main()
