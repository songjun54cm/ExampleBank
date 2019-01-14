#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import datetime
import argparse
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.metrics import roc_auc_score
# from Config import *
import tensorflow as tf

feature_size = 427
embedding_size = 10
deep_layers = [400, 200, 64]
deep_layers_activation = tf.nn.relu
batch_size = 256
learning_rate = 0.001
optimizer_type = "adam"
batch_norm_Flag = False
batch_norm_decay = 0.995
verbose = False
random_seed = 201810
l2_reg = 0.0
greater_is_better = True
epoch_number = 10000000


def get_file_name_list(basedir):
    dir_file_name_list = []
    if basedir.startswith('hdfs://'):
        filenames = [f.split()[-1] for f in os.popen('hadoop fs -ls ' + basedir)]
    else:
        filenames = [os.path.join(basedir, f) for f in os.listdir(basedir)]

    for f in filenames:
        if 'part' in f and not f.endswith("_COPYING_") and not f.endswith("SUCCESS"):
            dir_file_name_list.append(f)
    return dir_file_name_list


def parse_data_func(batch_serialized_example):
    data = {
        "features": tf.FixedLenFeature([feature_size], tf.float32),
        "label": tf.FixedLenFeature([1], tf.float32)
    }
    data_batch = tf.parse_example(batch_serialized_example, features=data)
    return data_batch


def parse_test_data_func(batch_serialized_example):
    data = {
        "features": tf.FixedLenFeature([feature_size], tf.float32),
        "label": tf.FixedLenFeature([1], tf.float32),
        "playTime": tf.FixedLenFeature([1], tf.float32),
        "duration": tf.FixedLenFeature([1], tf.float32),
        "playPercent": tf.FixedLenFeature([1], tf.float32)
    }
    data_batch = tf.parse_example(batch_serialized_example, features=data)
    return data_batch


def parse_pred_data_func(batch_serialized_example):
    data = {
        "userId": tf.FixedLenFeature([1], tf.string),
        "songId": tf.FixedLenFeature([1], tf.string),
        "features": tf.FixedLenFeature([feature_size], tf.float32),
        "srcid": tf.FixedLenFeature([1], tf.string),
        "aid": tf.FixedLenFeature([1], tf.string),
        "algtype": tf.FixedLenFeature([1], tf.string)
    }
    data_batch = tf.parse_example(batch_serialized_example, features=data)
    return data_batch


def read_train_data(config, num_parallel_reads=8, name=None):
    train_path = config.train
    if str(config.mode) == "train":
        repeat = 3
    else:
        repeat = int(config.repeat)
    train_files = get_file_name_list(train_path)
    train = tf.data.TFRecordDataset(train_files, num_parallel_reads=num_parallel_reads) \
        .repeat(repeat) \
        .batch(batch_size=batch_size) \
        .shuffle(buffer_size=10) \
        .map(lambda x: parse_data_func(x)) \
        .prefetch(buffer_size=2)

    return train.make_one_shot_iterator().get_next(name=name)


def read_dev_data(dev_path, repeat_num, num_parallel_reads=1, name=None):
    dev_files = get_file_name_list(dev_path)
    dev = tf.data.TFRecordDataset(dev_files, num_parallel_reads=num_parallel_reads) \
        .batch(batch_size=batch_size) \
        .repeat(repeat_num) \
        .map(lambda x: parse_data_func(x)) \
        .prefetch(buffer_size=1)

    return dev.make_one_shot_iterator().get_next(name=name)


def read_test_data(test_path, repeat_num, num_parallel_reads=1, name=None):
    dev_files = get_file_name_list(test_path)
    dev = tf.data.TFRecordDataset(dev_files, num_parallel_reads=num_parallel_reads) \
        .batch(batch_size=batch_size) \
        .repeat(repeat_num) \
        .map(lambda x: parse_test_data_func(x)) \
        .prefetch(buffer_size=1)

    return dev.make_one_shot_iterator().get_next(name=name)


def read_pred_data(config, num_parallel_reads=8, name=None):
    dir_file_name_list = []
    index = int(config.index)
    part_size = int(config.part_size)
    filenames = [f.split()[-1] for f in os.popen('hadoop fs -ls ' + config.input)]
    for f in filenames:
        if 'part' in f and not f.endswith("_COPYING_") and not f.endswith("SUCCESS"):
            dir_file_name_list.append(f)
    pred_data = dir_file_name_list[index * part_size:(index + 1) * part_size]

    pred = tf.data.TFRecordDataset(pred_data, num_parallel_reads=num_parallel_reads) \
        .batch(batch_size=batch_size) \
        .repeat(1) \
        .map(lambda x: parse_pred_data_func(x)) \
        .prefetch(buffer_size=1)

    return pred.make_one_shot_iterator().get_next(name=name)


def initialize_weights():
    weights = dict()

    # embeddings
    weights["embeddings"] = tf.Variable(
        tf.random_normal([feature_size, embedding_size], 0.0, 0.01),
        name="w_embeddings")  # N * K

    # deep layers
    num_layer = len(deep_layers)
    dnn0_input_size = feature_size  # N
    glorot = np.sqrt(2.0 / (dnn0_input_size + deep_layers[0]))
    weights["layer_0"] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(dnn0_input_size, deep_layers[0])),
        dtype=np.float32, name="w_layer_0")
    weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[0])),
                                    dtype=np.float32, name="w_bias_0")  # 1 * layers[0]
    for i in range(1, num_layer):
        glorot = np.sqrt(2.0 / (deep_layers[i - 1] + deep_layers[i]))
        weights["layer_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(deep_layers[i - 1], deep_layers[i])),
            dtype=np.float32, name="w_layer_%d" % i)  # layers[i-1] * layers[i]
        weights["bias_%d" % i] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[i])),
            dtype=np.float32, name="w_bias_%d" % i)  # 1 * layer[i]

    # final concat projection layer
    concat_input_size = embedding_size + deep_layers[-1]
    glorot = np.sqrt(2.0 / (concat_input_size + 1))
    weights["concat_projection"] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(concat_input_size, 1)),
        dtype=np.float32, name="w_concat_projection")  # layers[i-1]*layers[i]
    weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32, name="w_concat_bias")

    return weights


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = batch_norm(x, decay=batch_norm_decay, center=True, scale=True, updates_collections=None,
                          is_training=True, reuse=None, trainable=True, scope=scope_bn)
    bn_inference = batch_norm(x, decay=batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=False, reuse=True, trainable=True, scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z


def train(config):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    inputs = tf.placeholder(tf.float32, shape=[None, feature_size], name="input")  # None * N
    labels = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
    train_phase = tf.placeholder(tf.bool, name="train_phase")

    weights = initialize_weights()

    # ---------- model ----------
    reshaped_input = tf.reshape(inputs, shape=[-1, feature_size, 1])  # None * N * 1
    embeddings = tf.multiply(weights["embeddings"], reshaped_input, name="embeddings")  # None * N * K

    # ---first order term ---
    # y_first_order = inputs  # None * N

    # --- second order term ---
    # sum_square part
    summed_features_emb = tf.reduce_sum(embeddings, 1)  # None * K
    summed_features_emb_square = tf.square(summed_features_emb)  # None * K

    # square_sum part
    squared_features_emb = tf.square(embeddings)  # None * N * K
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

    # second order
    y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K

    # --- Deep component ---
    # y_deep = tf.reshape(embeddings, shape=[-1, feature_size * embedding_size])  # None * (N*K)
    y_deep = inputs
    for i in range(0, len(deep_layers)):
        y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" % i]), weights["bias_%d" % i])  # None * layer[i] * 1
        if batch_norm_Flag:
            y_deep = batch_norm_layer(y_deep, train_phase=train_phase, scope_bn="bn_%d" % i)  # None * layer[i] * 1
        y_deep = deep_layers_activation(y_deep)

    # --- DeepFM ---
    concat_input = tf.concat([y_second_order, y_deep], axis=1)
    logits = tf.add(tf.matmul(concat_input, weights["concat_projection"]), weights["concat_bias"])

    # --- loss ---
    score = tf.nn.sigmoid(logits)
    loss = tf.losses.log_loss(labels, score)
    if l2_reg > 0:
        loss += tf.contrib.layers.l2_regularizer(l2_reg)(weights["concat_projection"])
        for i in range(len(deep_layers)):
            loss += tf.contrib.layers.l2_regularizer(l2_reg)(weights["layer_%d" % i])

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)

    pred_score = tf.nn.sigmoid(logits, name="score")
    prediction = tf.cast(tf.greater_equal(pred_score, 0.5), tf.float32, name="prediction")
    accuracy = tf.multiply(tf.reduce_sum(tf.cast(tf.equal(prediction, tf.cast(labels, tf.float32)), tf.float32)),
                           1.0 / batch_size, name="accuracy")

    # ---------- model ----------

    next_train_batch = read_train_data(config, num_parallel_reads=8, name="train")
    next_dev_batch = read_dev_data(config.dev, 20, num_parallel_reads=1, name="dev")

    ckpt_dir = os.path.join('../checkpoint', config.model)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)

        all_dev_label = list()
        all_dev_score = list()

        startTime = time.mktime(time.localtime())
        print("-------------------------------------")
        for i in xrange(1, epoch_number + 1):
            try:
                train_batch = sess.run(next_train_batch)
            except tf.errors.OutOfRangeError:
                print("train all data!")
                break

            sess.run(train_op, feed_dict={
                inputs: train_batch["features"],
                labels: train_batch["label"],
                train_phase: True
            })
            if i % 500 == 0:
                res1 = sess.run([score, accuracy, loss, weights["concat_bias"]],
                                feed_dict={inputs: train_batch["features"], labels: train_batch["label"],
                                           train_phase: False})
                s = [k[0] for k in res1[0]]
                l = [k[0] for k in train_batch["label"]]
                auc = roc_auc_score(l, s)
                print("Train--%d--Accuracy:" % i, round(res1[1], 8), "loss:", round(res1[2], 8),
                      "AUC:", round(auc, 8), "concat_bias:", round(res1[3], 8))
            if i % 500 == 0:
                dev_batch = sess.run(next_dev_batch)
                res2 = sess.run([score, accuracy, loss],
                                feed_dict={inputs: dev_batch["features"], labels: dev_batch["label"],
                                           train_phase: False})
                l = [k[0] for k in dev_batch["label"]]
                s = [k[0] for k in res2[0]]
                all_dev_label.extend(l)
                all_dev_score.extend(s)
                auc = roc_auc_score(l, s)
                print("Test--%d--Accuracy:" % i, round(res2[1], 8), "loss:", round(res2[2], 8), "AUC:", round(auc, 8))
            if i % 1000 == 0:
                saver.save(sess, ckpt_dir + '/v' + config.version, global_step=i)
                print("concat_projection: %s" % round(res1[3], 8))

        endTime = time.mktime(time.localtime())
        print("Finished, time consumed: %s hours %s minutes %s seconds." % (
            str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
            str((endTime - startTime) % 60)))
        all_dev_auc = roc_auc_score(all_dev_label, all_dev_score)
        print("All Dev AUC = %f" % all_dev_auc)


def train2(config):
    g = tf.get_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    inputs = tf.placeholder(tf.float32, shape=[None, feature_size], name="input")  # None * N
    labels = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
    train_phase = tf.placeholder(tf.bool, name="train_phase")

    weights = initialize_weights()

    # ---------- model ----------
    reshaped_input = tf.reshape(inputs, shape=[-1, feature_size, 1])  # None * N * 1
    embeddings = tf.multiply(weights["embeddings"], reshaped_input, name="embeddings")  # None * N * K

    # ---first order term ---
    # y_first_order = inputs  # None * N

    # --- second order term ---
    # sum_square part
    summed_features_emb = tf.reduce_sum(embeddings, 1)  # None * K
    summed_features_emb_square = tf.square(summed_features_emb)  # None * K

    # square_sum part
    squared_features_emb = tf.square(embeddings)  # None * N * K
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

    # second order
    y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K

    # --- Deep component ---
    # y_deep = tf.reshape(embeddings, shape=[-1, feature_size * embedding_size])  # None * (N*K)
    y_deep = inputs
    for i in range(0, len(deep_layers)):
        y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" % i]), weights["bias_%d" % i])  # None * layer[i] * 1
        if batch_norm_Flag:
            y_deep = batch_norm_layer(y_deep, train_phase=train_phase, scope_bn="bn_%d" % i)  # None * layer[i] * 1
        y_deep = deep_layers_activation(y_deep)

    # --- DeepFM ---
    concat_input = tf.concat([y_second_order, y_deep], axis=1)
    logits = tf.add(tf.matmul(concat_input, weights["concat_projection"]), weights["concat_bias"])

    # --- loss ---
    score = tf.nn.sigmoid(logits)
    loss = tf.losses.log_loss(labels, score)
    if l2_reg > 0:
        loss += tf.contrib.layers.l2_regularizer(l2_reg)(weights["concat_projection"])
        for i in range(len(deep_layers)):
            loss += tf.contrib.layers.l2_regularizer(l2_reg)(weights["layer_%d" % i])

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)

    pred_score = tf.nn.sigmoid(logits, name="score")
    prediction = tf.cast(tf.greater_equal(pred_score, 0.5), tf.float32, name="prediction")
    accuracy = tf.multiply(tf.reduce_sum(tf.cast(tf.equal(prediction, tf.cast(labels, tf.float32)), tf.float32)),
                           1.0 / batch_size, name="accuracy")

    next_train_batch = read_train_data(config, num_parallel_reads=8, name="train")
    next_dev_batch = read_dev_data(config.dev, 20, num_parallel_reads=1, name="dev")

    with g.as_default():
        ckpt_dir = os.path.join('../checkpoint', config.model)
        saver = tf.train.Saver(max_to_keep=10)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    with tf.Session(graph=g, config=tf_config) as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        all_dev_label = list()
        all_dev_score = list()

        startTime = time.mktime(time.localtime())
        print("-------------------------------------")
        for i in xrange(1, epoch_number + 1):
            try:
                train_batch = sess.run(next_train_batch)
            except tf.errors.OutOfRangeError:
                print("train all data!")
                break

            sess.run(train_op, feed_dict={
                inputs: train_batch["features"],
                labels: train_batch["label"],
                train_phase: True
            })
            if i % 500 == 0:
                res1 = sess.run([score, accuracy, loss],
                                feed_dict={inputs: train_batch["features"], labels: train_batch["label"],
                                           train_phase: False})
                s = [k[0] for k in res1[0]]
                l = [k[0] for k in train_batch["label"]]
                auc = roc_auc_score(l, s)
                print("Train--%d--Accuracy:" % i, round(res1[1], 8), "loss:", round(res1[2], 8), "AUC:", round(auc, 8))
            if i % 500 == 0:
                dev_batch = sess.run(next_dev_batch)
                res2 = sess.run([score, accuracy, loss],
                                feed_dict={inputs: dev_batch["features"], labels: dev_batch["label"],
                                           train_phase: False})
                l = [k[0] for k in dev_batch["label"]]
                s = [k[0] for k in res2[0]]
                all_dev_label.extend(l)
                all_dev_score.extend(s)
                auc = roc_auc_score(l, s)
                print("Test--%d--Accuracy:" % i, round(res2[1], 8), "loss:", round(res2[2], 8), "AUC:", round(auc, 8))
            if i % 1000 == 0:
                saver.save(sess, ckpt_dir + '/v' + config.version, global_step=i)

        endTime = time.mktime(time.localtime())
        print("Finished, time consumed: %s hours %s minutes %s seconds." % (
            str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
            str((endTime - startTime) % 60)))
        all_dev_auc = roc_auc_score(all_dev_label, all_dev_score)
        print("All Dev AUC = %f" % all_dev_auc)


def test2(config):
    g = tf.get_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    with g.as_default():
        next_test_batch = read_dev_data(config.test, 1, num_parallel_reads=8, name="test")
        ckpt_dir = os.path.join('../checkpoint', config.model)
        model_file = tf.train.latest_checkpoint(ckpt_dir)
        saver = tf.train.import_meta_graph(model_file + '.meta')

    with tf.Session(graph=g, config=tf_config) as sess:
        saver.restore(sess, model_file)
        scores = g.get_tensor_by_name("score:0")

        all_label = list()
        all_score = list()

        print("-------------------------------------")
        count = 0
        startTime = time.mktime(time.localtime())
        while True:
            try:
                test_batch = sess.run(next_test_batch)
            except tf.errors.OutOfRangeError:
                print("test all data!")
                break

            count += 1
            batch_score = sess.run(scores, feed_dict={
                "input:0": test_batch["features"], "label:0": test_batch["label"], "train_phase:0": False})
            l = [k[0] for k in test_batch["label"]]
            s = [k[0] for k in batch_score]
            all_label.extend(l)
            all_score.extend(s)
            if count % 10 == 0:
                auc = roc_auc_score(l, s)
                print("Test--%d--AUC = %f" % (count, auc))

        endTime = time.mktime(time.localtime())
        print("Finished, time consumed: %s hours %s minutes %s seconds." % (
            str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
            str((endTime - startTime) % 60)))
        all_auc = roc_auc_score(all_label, all_score)
        print("All Test AUC = %f" % all_auc)

    sess.close()
    return


def test(config):
    g = tf.get_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    with g.as_default():
        next_test_batch = read_test_data(config.test, 1, num_parallel_reads=8, name="test")
        ckpt_dir = os.path.join('../checkpoint', config.model)
        model_file = tf.train.latest_checkpoint(ckpt_dir)
        saver = tf.train.import_meta_graph(model_file + '.meta')

    with tf.Session(graph=g, config=tf_config) as sess:
        saver.restore(sess, model_file)
        scores = g.get_tensor_by_name("score:0")

        all_label = list()
        all_score = list()
        all_time_1 = list()
        all_time_2 = list()
        all_time_3 = list()
        all_pct_1 = list()
        all_pct_2 = list()
        all_pct_3 = list()

        print("-------------------------------------")
        count = 0
        startTime = time.mktime(time.localtime())
        while True:
            try:
                test_batch = sess.run(next_test_batch)
            except tf.errors.OutOfRangeError:
                print("test all data!")
                break

            count += 1
            batch_score = sess.run(scores, feed_dict={"input:0": test_batch["features"],
                                                      "label:0": test_batch["label"],
                                                      "train_phase:0": False})
            l = [k[0] for k in test_batch["label"]]
            t = [k[0] for k in test_batch["playTime"]]
            p = [k[0] for k in test_batch["playPercent"]]
            s = [k[0] for k in batch_score]
            all_label.extend(l)
            all_score.extend(s)

            z = zip(t, p, s)
            rank = sorted(z, key=lambda x: x[2], reverse=True)
            r1 = rank[:30]
            r2 = rank[30:60]
            r3 = rank[60:90]
            all_time_1.extend([x[0] for x in r1])
            all_time_2.extend([x[0] for x in r2])
            all_time_3.extend([x[0] for x in r3])
            all_pct_1.extend([x[1] for x in r1])
            all_pct_2.extend([x[1] for x in r2])
            all_pct_3.extend([x[1] for x in r3])

            if count % 100 == 0:
                try:
                    auc = roc_auc_score(l, s)
                    print("Test--%d--AUC = %f, "
                          "time_1 = %f, time_2 = %f, time_3 = %f, "
                          "pct_1 = %f, pct_2 = %f, pct_3 = %f" % (
                              count, auc,
                              np.mean([x[0] for x in r1]), np.mean([x[0] for x in r2]), np.mean([x[0] for x in r3]),
                              np.mean([x[1] for x in r1]), np.mean([x[1] for x in r2]), np.mean([x[1] for x in r3])
                          ))
                except:
                    print("only one label")

        endTime = time.mktime(time.localtime())
        print("Finished, time consumed: %s hours %s minutes %s seconds." % (
            str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
            str((endTime - startTime) % 60)))
        all_auc = roc_auc_score(all_label, all_score)
        print("All Test AUC = %f, "
              "time_1 = %f, time_2 = %f, time_3 = %f, "
              "pct_1 = %f, pct_2 = %f, pct_3 = %f" % (
                  all_auc,
                  np.mean(all_time_1), np.mean(all_time_2), np.mean(all_time_3),
                  np.mean(all_pct_1), np.mean(all_pct_2), np.mean(all_pct_3)
              ))

    sess.close()
    return


def prediction_ckpt(config):
    g = tf.get_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    with g.as_default():
        next_pred_batch = read_pred_data(config, num_parallel_reads=8, name="prediction")
        ckpt_dir = os.path.join('../checkpoint', config.model)
        model_file = tf.train.latest_checkpoint(ckpt_dir)
        saver = tf.train.import_meta_graph(model_file + '.meta')

    f = open(config.output, 'w')

    with tf.Session(graph=g, config=tf_config) as sess:
        saver.restore(sess, model_file)
        scores = g.get_tensor_by_name("score:0")

        print("-------------------------------------")
        startTime = time.mktime(time.localtime())
        print("startTime", startTime)
        while True:
            try:
                test_batch = sess.run(next_pred_batch)
            except tf.errors.OutOfRangeError:
                print("prediction all data!")
                break

            batch_score = sess.run(scores, feed_dict={
                "input:0": test_batch["features"], "label:0": test_batch["label"], "train_phase:0": False})
            u = [k[0] for k in test_batch["userId"]]
            s = [k[0] for k in test_batch["songId"]]
            p = [k[0] for k in batch_score]
            for i in range(len(u)):
                f.write(str(int(u[i])) + "\t" + str(int(s[i])) + "\t" + str(p[i]) + "\n")

        endTime = time.mktime(time.localtime())
        print("endTime", endTime)
        print("Finished, time consumed: %s hours %s minutes %s seconds." % (
            str(int((endTime - startTime) / 3600)), str(int((endTime - startTime) / 60)),
            str((endTime - startTime) % 60)))

    f.close()
    return


def prediction(config):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        pb_path = os.path.join('../pb', config.model) + ".pb"

        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        graph = tf.get_default_graph()
        next_pred_batch = read_pred_data(config, num_parallel_reads=8, name="prediction")

    f = open(config.output, 'w')
    with tf.Session(graph=graph, config=tf_config) as sess:
        scores = graph.get_tensor_by_name("score:0")
        print("-------------------------------------")
        startTime = time.mktime(time.localtime())
        print("startTime", startTime)
        while True:
            try:
                test_batch = sess.run(next_pred_batch)
            except tf.errors.OutOfRangeError:
                print("prediction all data!")
                break

            if batch_norm_Flag:
                batch_score = sess.run(scores, feed_dict={"input:0": test_batch["features"], "train_phase:0": False})
            else:
                batch_score = sess.run(scores, feed_dict={"input:0": test_batch["features"]})
            userId = [k[0] for k in test_batch["userId"]]
            songId = [k[0] for k in test_batch["songId"]]
            p = [str(k[0]) for k in batch_score]
            srcid = [k[0] for k in test_batch["srcid"]]
            aid = [k[0] for k in test_batch["aid"]]
            algtype = [k[0] for k in test_batch["algtype"]]
            for i in range(len(userId)):
                f.write("\t".join([userId[i], songId[i], p[i], srcid[i], aid[i], algtype[i]]) + "\n")

        endTime = time.mktime(time.localtime())
        print("endTime", endTime)
        print("Finished, time consumed: %s hours %s minutes %s seconds." % (
            str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
            str((endTime - startTime) % 60)))

    f.close()
    putCmd = "hadoop fs -put " + config.output + " " + (config.input)[:-3]
    res = os.popen(putCmd).readlines()
    print(res)
    os.popen("touch ../data/success_" + str(config.index)).readlines()
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', help='model version', default='1', required=True)
    parser.add_argument('--model', help='model name', default='v2', required=True)
    parser.add_argument('--mode', help='train or test or pred', required=True)
    parser.add_argument('--gpuid', help='gpu id to use', default='0', required=True)

    parser.add_argument('--index', help='index of index', default="0")  # pred
    parser.add_argument('--part_size', help='part_size', default="0")  # pred
    parser.add_argument('--input', help='hdfs path of tfrecord', default='v2')  # pred
    parser.add_argument('--output', help='local path of score txt', default='v2')  # pred

    parser.add_argument('--repeat', help='repeat number', default='v2')  # train2
    parser.add_argument('--train', help='hdfs path of train tfrecord', default='v2')  # train
    parser.add_argument('--dev', help='hdfs path of dev tfrecord', default='v2')  # train

    parser.add_argument('--test', help='hdfs path of test tfrecord', default='v2')  # test

    config = parser.parse_args()
    return config


def init_parameters(config):
    version = int(config.version)
    global feature_size, embedding_size, deep_layers, deep_layers_activation, batch_size
    global learning_rate, optimizer_type, batch_norm_Flag, batch_norm_decay, verbose
    global random_seed, l2_reg, greater_is_better, epoch_number

    if version == 1:
        batch_norm_Flag = False
        print("batch_norm_Flag", batch_norm_Flag)
    elif version == 2:
        batch_norm_Flag = True
        print("batch_norm_Flag", batch_norm_Flag)


if __name__ == '__main__':
    startTime = time.mktime(time.localtime())

    config = parse_args()
    init_parameters(config)

    mode = str(config.mode)

    if mode == "train":
        train(config)
    elif mode == "train2":
        train2(config)
    elif mode == "test":
        test(config)
    elif mode == "pred":
        prediction(config)
    else:
        print("unknow mode...")

    endTime = time.mktime(time.localtime())
    print("Finished, time consumed: %s hours %s minutes %s seconds." % (
        str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
        str((endTime - startTime) % 60)))
