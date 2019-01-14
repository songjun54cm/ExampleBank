import pandas as pd
import pickle
import os
from settings import DATA_DIR


fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
          'banner_pos', 'site_id','site_domain', 'site_category','app_id','app_domain',
          'app_category', 'device_model', 'device_type',
          'device_conn_type']

data = pd.read_csv('G://Datasets//avazuCTR//test.csv')

for field_name in fields:
    field_set_v = set(data[field_name].values)
    with open(os.path.join(DATA_DIR, "field2count/%s.pkl"%field_name), "rb") as f:
        field_set = pickle.load(f)
    difs = field_set_v.difference(field_set)
    print("field: %s, dif len: %d, diff: %s" % (field_name, len(difs), str(difs)))

