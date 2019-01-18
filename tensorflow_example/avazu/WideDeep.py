__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/18
import argparse
from absl import app
from absl import flags
import tensorflow as tf
import pickle
import os
import numpy as np
from settings import DATA_DIR
import json


def get_columns(fields_dict):
    def get_categorial_column(field_name):
        col = tf.feature_column.categorial_column_with_vocabulary_list(field_name, list(fields_dict[field_name].values()))
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


def train_input_fn():
    fields_name = ["id", "click", "hour", "C1", "banner_pos", "site_id", "site_domain", "site_category",
                   "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
                   "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
    def parse_csv(value):
        vals = tf.convert_to_tensor(value)
        vals[2] =


    train_data_path = 'G://Datasets//avazuCTR//train.csv'
    record_defaults = [[""]] * 24
    dataset = tf.contrib.data.CsvDataset(train_data_path, record_defaults, header=True)



def main(argv):
    fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
              'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
              'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
              'device_conn_type','click']

    fields_dict = {}
    for field in fields:
        with open(os.path.join(DATA_DIR, "dicts", "%s.pkl"%field), "rb") as f:
            fields_dict[field] = pickle.load(f)

    wide_column, deep_column = get_columns(fields_dict)


if __name__ == "__main__":
    app.run(main)