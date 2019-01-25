__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/22
import argparse
import os
from settings import DATA_DIR
import pickle
import pandas as pd

field_names = ['hour', 'C1', 'C15', 'C16', 'C18', 'C20',
                'banner_pos',  'site_category','app_category',
                'device_type','device_conn_type', 'C14','C17', 'C19', 'C21',
               'site_id','site_domain','app_id','app_domain',
               'device_model', 'device_id']

fields_dict = {}
for field in field_names:
    with open(os.path.join(DATA_DIR, "field2formField", "%s.pkl"%field), "rb") as f:
        fields_dict[field] = pickle.load(f)


def form_train_data():
    field_columns = ["id", "click", "hour", "C1", "banner_pos", "site_id", "site_domain", "site_category",
                     "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
                     "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
    train_data_path = 'G://Datasets//avazuCTR//train.csv'
    train_data = pd.read_csv(train_data_path, chunksize=20000, dtype=str)
    form_train_file_path = os.path.join('G://Datasets//avazuCTR//form_train.csv')
    form_train_file = open(form_train_file_path, "w")
    form_train_file.write(",".join(field_columns) + "\n")
    for data in train_data:
        for idx, row in data.iterrows():
            new_row = []
            for field in field_columns:
                v = row[field]
                if field not in field_names:
                    new_row.append(v)
                else:
                    if field == "hour":
                        v = v[-2:]
                    new_v = fields_dict[field][v]
                    new_row.append(new_v)
            form_train_file.write(",".join(new_row) + "\n")
            if idx % 10000 == 0:
                print("%d sample processed" % idx)
    form_train_file.close()


def form_valid_data():
    field_columns = ["id", "click", "hour", "C1", "banner_pos", "site_id", "site_domain", "site_category",
                     "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
                     "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
    valid_data_path = 'G://Datasets//avazuCTR//train_valid.csv'
    valid_data = pd.read_csv(valid_data_path, chunksize=20000, dtype=str)
    form_valid_file_path = os.path.join('G://Datasets//avazuCTR//form_train_valid.csv')
    form_valid_file = open(form_valid_file_path, "w")
    form_valid_file.write(",".join(field_columns) + "\n")
    for data in valid_data:
        for idx, row in data.iterrows():
            new_row = []
            for field in field_columns:
                v = row[field]
                if field not in field_names:
                    new_row.append(v)
                else:
                    if field == "hour":
                        v = v[-2:]
                    new_v = fields_dict[field][v]
                    new_row.append(new_v)
            form_valid_file.write(",".join(new_row) + "\n")
            if idx % 10000 == 0:
                print("%d sample processed" % idx)
    form_valid_file.close()


def form_test_data():
    field_columns = ["id", "hour", "C1", "banner_pos", "site_id", "site_domain", "site_category",
                     "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
                     "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]
    test_data_path = 'G://Datasets//avazuCTR//test.csv'
    test_data = pd.read_csv(test_data_path, chunksize=20000, dtype=str)
    form_test_file_path = os.path.join('G://Datasets//avazuCTR//form_test.csv')
    form_test_file = open(form_test_file_path, "w")
    form_test_file.write(",".join(field_columns) + "\n")
    for data in test_data:
        for idx, row in data.iterrows():
            new_row = []
            for field in field_columns:
                v = row[field]
                if field not in field_names:
                    new_row.append(v)
                else:
                    if field == "hour":
                        v = v[-2:]
                    new_v = fields_dict[field][v]
                    new_row.append(new_v)
            form_test_file.write(",".join(new_row) + "\n")
            if idx % 10000 == 0:
                print("%d sample processed" % idx)
    form_test_file.close()


if __name__ == "__main__":
    # form_train_data()
    form_valid_data()
    form_test_data()
