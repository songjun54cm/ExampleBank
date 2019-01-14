# -*- coding:utf-8 -*-

import tensorflow as tf
from sklearn.metrics import roc_auc_score

# ==========================================
dropout_fm = [1.0, 1.0]
dropout_deep = [0.5, 0.5, 0.5]
use_fm = True
use_deep = True
epoch = 10
batch_norm = 0
# -------------------------------------------
feature_size = 427
embedding_size = 5
deep_layers = [400, 200]
deep_layers_activation = tf.nn.relu
batch_size = 256
learning_rate = 0.001
optimizer_type = "adam"
train_phase = 0
batch_norm_decay = 0.995
verbose = False
random_seed = 201810
loss_type = "logloss"
eval_metric = roc_auc_score
l2_reg = 0.0
greater_is_better = True
epoch_number = 1000000
