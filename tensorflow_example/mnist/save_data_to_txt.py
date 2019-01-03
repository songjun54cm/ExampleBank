"""
Author: songjun
Date: 2018/4/13
Description:
Usage:
"""
import tensorflow as tf
import numpy as np
import os

print('loading data.')
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
print('train data shape %s' % str(train_data.shape))
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
print('train label shape %s' % str(train_labels.shape))
eval_data = mnist.test.images  # Returns np.array
# eval_data = np.reshape(eval_data, (eval_data.shape[0], 28, 28, 1))
print('eval data shape %s' % str(eval_data.shape))
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
print('eval labels shape %s' % str(eval_labels.shape))


if not os.path.exists('../data/mnist'): os.makedirs('../data/mnist/')
with open("../data/mnist/train_labels.txt", "w") as f:
    train_labels = [str(l) for l in train_labels.tolist()]
    f.write("\n".join(train_labels))

with open("../data/mnist/train_data.txt", "w") as f:
    for i in range(train_data.shape[0]):
        fea = [str(v) for v in train_data[i, :]]
        f.write(' '.join(fea))
        f.write('\n')

with open('../data/mnist/eval_labels.txt', 'w') as f:
    eval_labels = [str(l) for l in eval_labels.tolist()]
    f.write('\n'.join(eval_labels))

with open('../data/mnist/eval_data.txt', 'w') as f:
    for i in range(eval_data.shape[0]):
        fea = [str(v) for v in eval_data[i, :]]
        f.write(' '.join(fea))
        f.write('\n')

