__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/14
import argparse
import pickle
import os
from settings import DATA_DIR

direct_encoding_fields = ['hour', 'C1', 'C15', 'C16', 'C18', 'C20',
                          'banner_pos',  'site_category','app_category',
                          'device_type','device_conn_type']

frequency_encoding_fields = ['C14','C17', 'C19', 'C21',
                             'site_id','site_domain','app_id','app_domain',
                             'device_model', 'device_id']

ind = 0
total_field_dict = {}

# load direct encoding field
for field_name in direct_encoding_fields:
    with open(os.path.join(DATA_DIR, "field2set", "%s.pkl"%field_name), "rb") as f:
        field_set = pickle.load(f)
    field_dict = {}
    for value in list(field_set):
        field_dict[value] = ind
        ind += 1
    total_field_dict.update(field_dict)
    with open(os.path.join(DATA_DIR, "dicts", "%s.pkl"%field_name), 'wb') as f:
        pickle.dump(field_dict, f)

for field_name in frequency_encoding_fields:
    with open(os.path.join(DATA_DIR, "field2count", "%s.pkl"%field_name), "rb") as f:
        field2cnt = pickle.load(f)
    field_dict = {}
    ind_rare = None
    for k,cnt in field2cnt.items():
        if cnt < 10:
            if ind_rare is None:
                field_dict[k] = ind
                ind_rare = ind
                ind += 1
            else:
                field_dict[k] = ind_rare
        else:
            field_dict[k] = ind
            ind += 1
    total_field_dict.update(field_dict)
    with open(os.path.join(DATA_DIR, "dicts", "%s.pkl"%field_name), "wb") as f:
        pickle.dump(field_dict, f)

field_dict = {}
with open(os.path.join(DATA_DIR, "field2set", "click.pkl"), "rb") as f:
    click = pickle.load(f)
field_sets = click
for value in list(field_sets):
    field_dict[value] = ind
    ind += 1
with open(os.path.join(DATA_DIR, "dicts", "click.pkl"), 'wb') as f:
    pickle.dump(field_dict, f)
