# coding:utf-8
import numpy as np
import pandas as pd
import pickle
import os
from settings import DATA_DIR


def ffm_onehot_representation(sample, fields_dict, array_length):
    array = np.zeros([array_length])
    for field in fields_dict:
        if field == "click":
            continue
        if field == "hour":
            field_value = int(str(sample[field])[-2:])
        else:
            field_value = sample[field]
        ind = fields_dict[field][field_value]
        array[ind] = 1.0
    return array, label


def ffm_batch_data_generate(batch_data, fields_dict, array_length):
    batch_x = []
    batch_y = []
    for i in range(len(batch_data)):
        sample = batch_data.iloc[i,:]
        click = sample["click"]
        if click == 0:
            label = 0
        else:
            label = 1
        batch_y.append(label)
        array = ffm_onehot_representation(sample, fields_dict, array_length)
        batch_x.append(array)
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    return batch_x, batch_y


def one_hot_representation(sample, fields_dict, isample):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param isample: sample index
    :return: sample index
    """
    index = []
    for field in fields_dict:
        if field == "click":
            continue
        # get index of array
        if field == 'hour':
            field_value = int(str(sample[field])[-2:])
        else:
            field_value = sample[field]
        ind = fields_dict[field][field_value]
        index.append([isample, ind])
    return index

def train_batch_sparse_data_generate(batch_data, field_dict):
    labels = []
    indexes = []
    for i in range(len(batch_data)):
        sample = batch_data.iloc[i, :]
        click = sample["click"]
        if click == 0:
            label = 0
        else:
            label = 1
        labels.append(label)
        index = one_hot_representation(sample, field_dict, i)
        indexes.extend(index)
    return indexes, labels


def train_sparse_data_generate(train_data, field_dict):
    sparse_data = []
    ibatch = 0
    for data in train_data:
        indexes, labels = train_batch_sparse_data_generate(data, field_dict)
        sparse_data.append({"indexes":indexes, "labels":labels})
        ibatch += 1
        if ibatch % 1000 == 0:
            with open(os.path.join(DATA_DIR, "sparse_data", "train", "sparse_data_%d_%d.pkl"%(ibatch-1000, ibatch-1)), "wb") as f:
                pickle.dump(sparse_data, f)
            sparse_data = []
            print("%d batch has finished." % ibatch)
    with open(os.path.join(DATA_DIR, "sparse_data", "train", "sparse_data_%d_%d.pkl"%(ibatch-len(sparse_data), ibatch-1)), "wb") as f:
        pickle.dump(sparse_data, f)


def ttest_sparse_data_generate(batch_data, field_dict):
    ids = []
    indexes = []
    for i in range(len(batch_data)):
        sample = batch_data.iloc[i,:]
        ids.append(sample["id"])
        index = one_hot_representation(sample, field_dict, i)
        indexes.extend(index)
    return indexes, ids


def ttest_sparse_data_generate(test_data, fields_dict):
    sparse_data = []
    ibatch = 0
    for data in test_data:
        indexes, ids = ttest_sparse_data_generate(data, fields_dict)
        sparse_data.append({"indexes":indexes, "id":ids})
        ibatch += 1
        if ibatch % 1000 == 0:
            with open(os.path.join(DATA_DIR, "sparse_data", "test", "sparse_data_%d_%d.pkl"%(ibatch-1000, ibatch-1)), "wb") as f:
                pickle.dump(sparse_data, f)
            sparse_data = []
            print("%d batch has finished." % ibatch)
    with open(os.path.join(DATA_DIR, "sparse_data", "test", "sparse_data_%d_%d.pkl"%(ibatch-len(sparse_data), ibatch-1)), "wb") as f:
        pickle.dump(sparse_data, f)


if __name__ == '__main__':
    # fields_train = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
    #           'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
    #           'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
    #           'device_conn_type']  #,'click']
    #
    # fields_test = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
    #                'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
    #                'app_id', 'device_id', 'app_category', 'device_model', 'device_type',
    #                'device_conn_type']

    fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
              'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
              'app_id', 'device_id', 'app_category', 'device_model', 'device_type',
              'device_conn_type']

    batch_size = 512
    train = pd.read_csv('G://Datasets//avazuCTR//train.csv', chunksize=batch_size)

    test = pd.read_csv('G://Datasets//avazuCTR//test.csv', chunksize=batch_size)

    # loading dicts
    fields_dict = {}
    for field in fields:
        with open(os.path.join(DATA_DIR, 'dicts', field+'.pkl'),'rb') as f:
            fields_dict[field] = pickle.load(f)
            print("field: %s, len: %d" % (field, len(fields_dict[field])))

    train_sparse_data_generate(train, fields_dict)
    ttest_sparse_data_generate(test, fields_dict)
