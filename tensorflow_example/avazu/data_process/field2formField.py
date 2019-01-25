__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/22
import argparse
import pickle
import os
from settings import DATA_DIR


direct_encoding_fields = ['hour', 'C1', 'C15', 'C16', 'C18', 'C20',
                          'banner_pos',  'site_category','app_category',
                          'device_type','device_conn_type', "click"]

frequency_encoding_fields = ['C14','C17', 'C19', 'C21',
                             'site_id','site_domain','app_id','app_domain',
                             'device_model', 'device_id']

field2formField = {}
for field_name in direct_encoding_fields:
    with open(os.path.join(DATA_DIR, "field2set", "%s.pkl"%field_name), "rb") as f:
        field_set = pickle.load(f)
    field_dict = {}
    for v in list(field_set):
        field_dict[v] = v
    with open(os.path.join(DATA_DIR, "field2formField", "%s.pkl"%field_name), 'wb') as f:
        pickle.dump(field_dict, f)

for field_name in frequency_encoding_fields:
    with open(os.path.join(DATA_DIR, "field2count", "%s.pkl"%field_name), "rb") as f:
        field2cnt = pickle.load(f)
    field_dict = {}
    for k,cnt in field2cnt.items():
        if cnt < 10:
            field_dict[k] = "other"
        else:
            field_dict[k] = k
    with open(os.path.join(DATA_DIR, "field2formField", "%s.pkl"%field_name), "wb") as f:
        pickle.dump(field_dict, f)
