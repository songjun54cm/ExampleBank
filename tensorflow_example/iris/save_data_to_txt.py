__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/12/13
import argparse
import numpy as np
from sklearn.datasets import load_iris
import random
import os
import _pickle as pkl

print('loading data.')
x, y = load_iris(return_X_y=True)
samples = [(a,b) for (a,b) in zip(x,y)]
random.shuffle(samples)
train_samples = samples[:int(len(samples)*0.8)]
eval_samples = samples[int(len(samples)*0.8):]
train_fea = np.asarray([x[0] for x in train_samples])
train_label = np.asarray([x[1] for x in train_samples])
print('train data shape %s' % str(train_fea.shape))
print('train label shape %s' % str(train_label.shape))
eval_fea = np.asarray([x[0] for x in eval_samples])
eval_label = np.asarray([x[1] for x in eval_samples])
print('eval data shape %s' % str(eval_fea.shape))
print('eval labels shape %s' % str(eval_label.shape))

data_path = "../data/iris/data.pkl"
data = {
    "train_fea": train_fea,
    "train_label": train_label,
    "eval_fea": eval_fea,
    "eval_label": eval_label
}
pkl.dump(data, open(data_path, "wb"))


if not os.path.exists('../data/iris'): os.makedirs('../data/iris/')
with open('../data/iris/eval_labels.txt', 'w') as f:
    eval_labels = [str(l) for l in eval_label.tolist()]
    f.write('\n'.join(eval_labels))

with open('../data/iris/eval_data.txt', 'w') as f:
    for i in range(eval_fea.shape[0]):
        fea = [str(v) for v in eval_fea[i, :]]
        f.write(' '.join(fea))
        f.write('\n')

if not os.path.exists('../data/iris'): os.makedirs('../data/iris/')
with open('../data/iris/train_labels.txt', 'w') as f:
    labels = [str(l) for l in train_label.tolist()]
    f.write('\n'.join(labels))

with open('../data/iris/train_data.txt', 'w') as f:
    for i in range(train_fea.shape[0]):
        fea = [str(v) for v in train_fea[i, :]]
        f.write(' '.join(fea))
        f.write('\n')