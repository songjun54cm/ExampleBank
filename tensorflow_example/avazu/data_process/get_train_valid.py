__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/22
import pandas as pd
import os
from settings import DATA_DIR
import random

train_file = 'G://Datasets//avazuCTR//train.csv'
out_file = 'G://Datasets//avazuCTR//train_valid.csv'
out_f = open(out_file, "w")
with open(train_file, "r") as f:
    line = f.readline()
    out_f.write(line)
    line = f.readline()
    while line:
        if random.random() < 0.001:
            out_f.write(line)
        line = f.readline()

