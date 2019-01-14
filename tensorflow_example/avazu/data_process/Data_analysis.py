import pandas as pd
import pickle
from collections import Counter
from settings import DATA_DIR
import os

train = pd.read_csv('G://Datasets//avazuCTR//train.csv',chunksize=20000)
test = pd.read_csv('G://Datasets//avazuCTR//test.csv',chunksize=20000)

field_set = {
    "C14": dict(),
    "C17": dict(),
    "C19": dict(),
    "C21": dict(),
    "site_id": dict(),
    "site_domain": dict(),
    "app_id": dict(),
    "app_domain": dict(),
    "device_model": dict(),
    "device_id": dict(),
    "device_ip": dict(),

    "hour": dict(),
    "C1": dict(),
    "C15": dict(),
    "C16": dict(),
    "C18": dict(),
    "C20": dict(),
    "banner_pos": dict(),
    "site_category": dict(),
    "app_category": dict(),
    "device_type": dict(),
    "device_conn_type": dict()
}
click = dict()

count = 0
for data in train:
    for field_name in field_set.keys():
        field_list = data[field_name].values
        for k,v in Counter(field_list).items():
            if k in field_set[field_name].keys():
                field_set[field_name][k] += v
            else:
                field_set[field_name][k] = v

    click_list = data["click"].values
    for k,v in Counter(click_list).items():
        if k in click.keys():
            click[k] += v
        else:
            click[k] = v

    count += 1
    if count % 100 == 0:
        print('{} has finished'.format(count))


count = 0
for data in test:
    for field_name in field_set.keys():
        field_list = data[field_name].values
        for k,v in Counter(field_list).items():
            if k in field_set[field_name].keys():
                field_set[field_name][k] += v
            else:
                field_set[field_name][k] = v

    count += 1
    if count % 100 == 0:
        print('{} has finished'.format(count))

# save dictionaries
for field_name in field_set.keys():
    with open(os.path.join(DATA_DIR, "field2count/%s.pkl"%field_name), "wb") as f:
        field_dict = field_set[field_name]
        print("field: %s, size: %d" % (field_name, len(field_dict)))
        pickle.dump(field_dict, f)

with open(os.path.join(DATA_DIR, "field2count/click.pkl"), "wb") as f:
    field_dict = click
    print("field: %s, size: %d" % ("click", len(field_dict)))
    pickle.dump(field_dict, f)

