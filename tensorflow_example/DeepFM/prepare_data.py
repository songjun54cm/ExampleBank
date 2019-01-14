__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/10
import argparse
import numpy as np
import pandas as pd
import os
import _pickle as pkl
import logging
from collections import Counter


def gen_dict_pkl():
    data_dir = "G://Datasets//avazuCTR"
    train_csv = os.path.join(data_dir, "train.csv")

    # for site_id, site_domain, app_id, app_domain, device_model,
    # device_ip, device_id fields,C14,C17,C19,C21, one-hot using frequency
    # for other fields, one-hot-encoding directly

    # one-hot encoding directly
    click = set()
    hour = set()
    C1 = set()
    banner_pos = set()
    site_category = set()
    app_category = set()
    device_type = set()
    device_conn_type = set()
    C15 = set()
    C16 = set()
    C18 = set()
    C20 = set()
    hours = set(range(24))

    # one-hot encoding by frequency bucket
    C14 = []
    C17 = []
    C19 = []
    C21 = []
    site_id = []
    site_domain = []
    app_id = []
    app_domain = []
    device_model = []
    device_ip = []
    device_id = []

    train = pd.read_csv(train_csv)
    for data in train:
        click_v = set(data['click'].values)
        click = click | click_v

        C1_v = set(data['C1'].values)
        C1 = C1 | C1_v

        C15_v = set(data['C15'].values)
        C15 = C15 | C15_v

        C16_v = set(data['C16'].values)
        C16 = C16 | C16_v

        C18_v = set(data['C18'].values)
        C18 = C18 | C18_v

        C20_v = set(data['C20'].values)
        C20 = C20 | C20_v

        banner_pos_v = set(data['banner_pos'].values)
        banner_pos = banner_pos | banner_pos_v

        site_category_v = set(data['site_category'].values)
        site_category = site_category | site_category_v

        app_category_v = set(data['app_category'].values)
        app_category = app_category | app_category_v

        device_type_v = set(data['device_type'].values)
        device_type = device_type | device_type_v

        device_conn_type_v = set(data['device_conn_type'].values)
        device_conn_type = device_conn_type | device_conn_type_v

    # save dictionaries
    res = {
        "click": click,
        "hour": hours,
        "C1": C1,
        "C15": C15,
        "C16": C16,
        "C18": C18,
        "C20": C20,
        "banner_pos": banner_pos,
        "site_category": site_category,
        "app_category": app_category,
        "device_type": device_type,
        "device_conn_type": device_conn_type
    }
    with open('data/avazu/avazu_dict.pkl', 'wb') as f:
        pkl.dump(res, f)
    return res


def main(config):
    fea_dicts = gen_dict_pkl()

    direct_encoding_fields = ['hour', 'C1', 'C15', 'C16', 'C18', 'C20',
                              'banner_pos',  'site_category','app_category',
                              'device_type','device_conn_type']

    frequency_encoding_fields = ['C14','C17', 'C19', 'C21',
                                 'site_id','site_domain','app_id','app_domain',
                                 'device_model', 'device_id']



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)