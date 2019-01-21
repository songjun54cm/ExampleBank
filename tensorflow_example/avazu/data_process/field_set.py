__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/14
import argparse
from settings import DATA_DIR
import pickle
import os
f2c_dir = os.path.join(DATA_DIR, 'field2count')

field_list = ["click", "C1", "banner_pos", "site_category",
              "app_category", "device_type", "device_conn_type",
              "C15", "C16", "C18", "C20", 'hour',

              "C14", "C17", "C19", "C21",
              "site_id", "site_domain", "app_id", "app_domain",
              "device_model", "device_id"]

# hours = set(list(range(24)))

for field_name in field_list:
    with open(os.path.join(f2c_dir, "%s.pkl" % field_name), 'rb') as f:
        field_cnt = pickle.load(f)
    keys = set(field_cnt.keys())
    with open(os.path.join(DATA_DIR, "field2set", "%s.pkl"%field_name), "wb") as f:
        pickle.dump(keys, f)

# with open(os.path.join(DATA_DIR, "field2set", "hour.pkl"), "wb") as f:
#     pickle.dump(hours, f)
