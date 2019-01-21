__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/14
import argparse
import pandas as pd
import os
from settings import DATA_DIR

train = pd.read_csv('G://Datasets//avazuCTR//train.csv', chunksize=1000)
for data in train:
    data.to_csv(os.path.join(DATA_DIR, 'train_debug.csv'), index=False)
    break

test = pd.read_csv('G://Datasets//avazuCTR//test.csv', chunksize=1000)
for data in test:
    data.to_csv(os.path.join(DATA_DIR, 'test_debug.csv'), index=False)
    break
