import matplotlib.pyplot as plt
import pickle
from collections import Counter
import pandas as pd
# load field2count dictionaries
import os
from settings import DATA_DIR
f2c_dir = os.path.join(DATA_DIR, 'field2count')
field_list = ["C14", "C17", "C19", "C21", "site_id",
              "site_domain", "app_id", "app_domain",
              "device_model", "device_id", "device_ip",
              "hour", "C1", "C15", "C16", "C18", "C20",
              "banner_pos", "site_category", "app_category",
              "device_type", "device_conn_type", "click"]

for field_name in field_list:
    with open(os.path.join(f2c_dir, "%s.pkl" % field_name), 'rb') as f:
        field_cnt = pickle.load(f)
    cnts = field_cnt.values()
    b = sorted(cnts, reverse=True)
    frequency = list(range(1, len(cnts)+1))
    count = b

    # result = Counter(field_cnt.values())
    # b = sorted(result.items(), key=lambda x:float(x[1]) + (1.0 / float(x[0])), reverse=True)
    # frequency = []
    # [frequency.append(x[0]) for x in b]
    # count = []
    # [count.append(x[1]) for x in b]

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(frequency, count, 'o-', color = 'blue')
    ax1.set_xscale('log')
    ax1.grid(True, which='major', axis='both')
    ax1.set_xlabel('frequency(%d)'%len(frequency),fontsize = 15)
    ax1.set_ylabel('counts', fontsize = 15)
    ax1.set_title('field of %s' % field_name, fontsize=15)
    plt.savefig(os.path.join(f2c_dir, field_name))
    plt.close()
    print("%s finish." % field_name)
