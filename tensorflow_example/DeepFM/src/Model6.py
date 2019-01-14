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

feature_size = 436
embedding_size = 10
deep_layers = [400, 200, 64]
deep_layers_activation = tf.nn.relu
batch_size = 256
learning_rate = 0.001
optimizer_type = "adam"
batch_norm_Flag = False
batch_norm_decay = 0.995
verbose = False
random_seed = 201901
l2_reg = 0.0
epoch_number = 900000


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
        "label": tf.FixedLenFeature([1], tf.float32),
        "playTime": tf.FixedLenFeature([1], tf.float32),
        "duration": tf.FixedLenFeature([1], tf.float32),
        "playPercent": tf.FixedLenFeature([1], tf.float32)
    }
    data_batch = tf.parse_example(batch_serialized_example, features=data)
    return data_batch


def read_train_data(config, num_parallel_reads=8, name=None):
    train_path = config.train
    if str(config.mode) == "train":
        repeat = 2
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
    weights["concat_projection1"] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(concat_input_size, 1)),
        dtype=np.float32, name="w_concat_projection1")  # layers[i-1]*layers[i]
    weights["concat_projection2"] = tf.Variable(
        np.random.normal(loc=0, scale=glorot, size=(concat_input_size, 1)),
        dtype=np.float32, name="w_concat_projection2")  # layers[i-1]*layers[i]
    weights["concat_bias1"] = tf.Variable(tf.constant(0.01), dtype=np.float32, name="w_concat_bias1")
    weights["concat_bias2"] = tf.Variable(tf.constant(0.01), dtype=np.float32, name="w_concat_bias2")

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
    playTime = tf.placeholder(tf.float32, shape=[None, 1], name="playTime")  # None * 1
    duration = tf.placeholder(tf.float32, shape=[None, 1], name="duration")  # None * 1
    playPercent = tf.placeholder(tf.float32, shape=[None, 1], name="playPercent")  # None * 1
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
    logits1 = tf.add(tf.matmul(concat_input, weights["concat_projection1"]), weights["concat_bias1"])
    logits2 = tf.add(tf.matmul(concat_input, weights["concat_projection2"]), weights["concat_bias2"])

    # --- loss ---
    sub_score = tf.nn.sigmoid(logits1, name="score1")
    loss1 = tf.losses.log_loss(labels, sub_score)

    play_score = tf.nn.sigmoid(logits2, name="score2")

    if int(config.version) in [2]:
        loss2 = tf.reduce_mean(tf.square(playPercent - play_score))
    else:
        playend = tf.cast(tf.greater_equal(tf.subtract(playTime, duration), -1), tf.float32, name="playend")
        loss2 = tf.losses.log_loss(playend, play_score)

    loss = tf.add(loss1, loss2)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)

    prediction = tf.cast(tf.greater_equal(sub_score, 0.5), tf.float32, name="prediction")
    accuracy = tf.multiply(tf.reduce_sum(tf.cast(tf.equal(prediction, tf.cast(labels, tf.float32)), tf.float32)),
                           1.0 / batch_size, name="accuracy")

    # ---------- model ----------

    next_train_batch = read_train_data(config, num_parallel_reads=8, name="train")
    # next_dev_batch = read_dev_data(config.dev, 20, num_parallel_reads=1, name="dev")

    ckpt_dir = os.path.join('../checkpoint', config.model)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)

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
                playTime: train_batch["playTime"],
                duration: train_batch["duration"],
                playPercent: train_batch["playPercent"],
                train_phase: True
            })
            if i % 500 == 0:
                res1 = sess.run([sub_score, accuracy, loss1, loss2, loss],
                                feed_dict={inputs: train_batch["features"],
                                           labels: train_batch["label"],
                                           playTime: train_batch["playTime"],
                                           duration: train_batch["duration"],
                                           playPercent: train_batch["playPercent"],
                                           train_phase: False})
                s = [k[0] for k in res1[0]]
                l = [k[0] for k in train_batch["label"]]
                auc = roc_auc_score(l, s)
                # print(zip(s, l))
                print("Train--%d--Accuracy:" % i, round(res1[1], 6),
                      "loss1:", round(res1[2], 6), "loss2:", round(res1[3], 6),
                      "loss:", round(res1[4], 6), "AUC:", round(auc, 6))
            # if i % 500 == 0:
            #     dev_batch = sess.run(next_dev_batch)
            #     res2 = sess.run([sub_score, accuracy, loss1, loss2, loss],
            #                     feed_dict={inputs: dev_batch["features"],
            #                                labels: dev_batch["label"],
            #                                playTime: dev_batch["playTime"],
            #                                duration: dev_batch["duration"],
            #                                playPercent: train_batch["playPercent"],
            #                                train_phase: False})
            #     l = [k[0] for k in dev_batch["label"]]
            #     s = [k[0] for k in res2[0]]
            #     all_dev_label.extend(l)
            #     all_dev_score.extend(s)
            #     auc = roc_auc_score(l, s)
            #     print("Test---%d--Accuracy:" % i, round(res2[1], 6),
            #           "loss1:", round(res2[2], 6), "loss2:", round(res2[3], 6),
            #           "loss:", round(res2[4], 6), "AUC:", round(auc, 6))
            if i % 1000 == 0:
                saver.save(sess, ckpt_dir + '/v' + config.version, global_step=i)

        endTime = time.mktime(time.localtime())
        print("Finished, time consumed: %s hours %s minutes %s seconds." % (
            str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
            str((endTime - startTime) % 60)))
        # all_dev_auc = roc_auc_score(all_dev_label, all_dev_score)
        # print("All Dev AUC = %f" % all_dev_auc)


def test(config):
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
        sub_scores = g.get_tensor_by_name("score1:0")
        play_scores = g.get_tensor_by_name("score2:0")

        all_label1 = list()
        all_label2 = list()
        all_score1 = list()
        all_score2 = list()
        all_score3 = list()
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
            batch_score = sess.run([sub_scores, play_scores],
                                   feed_dict={"input:0": test_batch["features"],
                                              "label:0": test_batch["label"],
                                              "playTime:0": test_batch["playTime"],
                                              "duration:0": test_batch["duration"],
                                              "playPercent:0": test_batch["playPercent"],
                                              "train_phase:0": False})
            l1 = [k[0] for k in test_batch["label"]]
            t = [k[0] for k in test_batch["playTime"]]
            d = [k[0] for k in test_batch["duration"]]
            l2 = [1 if i - j >= -1 else 0 for (i, j) in zip(t, d)]
            p = [k[0] for k in test_batch["playPercent"]]
            s1 = [k[0] for k in batch_score[0]]
            s2 = [k[0] for k in batch_score[1]]
            # s3 = [i / 0.156 + j / 0.273 for (i, j) in zip(s1, s2)]
            # s3 = [0.66 * i + 0.34 * j for (i, j) in zip(s1, s2)]
            s3 = [0.64 * i + 0.36 * j for (i, j) in zip(s1, s2)]
            all_label1.extend(l1)
            all_label2.extend(l2)
            all_score1.extend(s1)
            all_score2.extend(s2)
            all_score3.extend(s3)

            z = zip(t, p, s3)
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
                    auc = roc_auc_score(l1, s3)
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
        all_auc1 = roc_auc_score(all_label1, all_score1)
        all_auc2 = roc_auc_score(all_label2, all_score2)
        all_auc3 = roc_auc_score(all_label1, all_score3)

        print("score1 avg:", np.mean(all_score1), "label1 avg:", np.mean(all_label1))
        print("score2 avg:", np.mean(all_score2), "label2 avg:", np.mean(all_label2))
        print("All Test  AUC1 = %f, AUC2 = %f, AUC3 = %f, "
              "time_1 = %.3f, time_2 = %.3f, time_3 = %.3f, "
              "pct_1 = %.3f, pct_2 = %.3f, pct_3 = %.3f" % (
                  all_auc1, all_auc2, all_auc3,
                  np.mean(all_time_1), np.mean(all_time_2), np.mean(all_time_3),
                  np.mean(all_pct_1), np.mean(all_pct_2), np.mean(all_pct_3)
              ))

    sess.close()
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', help='model version', default='1', required=True)
    parser.add_argument('--model', help='model name', default='v5', required=True)
    parser.add_argument('--mode', help='train or test or pred', required=True)
    parser.add_argument('--gpuid', help='gpu id to use', default='0', required=True)

    parser.add_argument('--repeat', help='repeat number', default='2')  # train2
    parser.add_argument('--train', help='hdfs path of train tfrecord', default='v5')  # train
    parser.add_argument('--dev', help='hdfs path of dev tfrecord', default='v5')  # train

    parser.add_argument('--test', help='hdfs path of test tfrecord', default='v5')  # test

    config = parser.parse_args()
    return config


def printParameter():
    print("feature_size     ", feature_size)
    print("embedding_size   ", embedding_size)
    print("deep_layers      ", deep_layers)
    print("batch_size       ", batch_size)
    print("learning_rate    ", learning_rate)
    print("batch_norm_Flag  ", batch_norm_Flag)
    print("batch_norm_decay ", batch_norm_decay)
    print("verbose          ", verbose)
    print("random_seed      ", random_seed)
    print("l2_reg           ", l2_reg)
    print("epoch_number     ", epoch_number)


def init_parameters(config):
    version = int(config.version)
    global feature_size, embedding_size, deep_layers, deep_layers_activation, batch_size
    global learning_rate, optimizer_type, batch_norm_Flag, batch_norm_decay, verbose
    global random_seed, l2_reg, epoch_number

    if version == 4:
        feature_size = 564
    if version == 5:
        feature_size = 467
    if version == 6:
        feature_size = 536
        deep_layers = [500, 256, 64]
    if version == 7:
        feature_size = 437

    if str(config.mode) == "test":
        batch_size = 200

    printParameter()


if __name__ == '__main__':
    startTime = time.mktime(time.localtime())

    config = parse_args()
    init_parameters(config)

    mode = str(config.mode)
    version = int(config.version)

    if mode == "train":
        train(config)
    elif mode == "test":
        test(config)
    else:
        print("unknow mode...")

    endTime = time.mktime(time.localtime())
    print("Finished, time consumed: %s hours %s minutes %s seconds." % (
        str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
        str((endTime - startTime) % 60)))
