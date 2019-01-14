#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import argparse
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.metrics import roc_auc_score
# from Config import *
import tensorflow as tf

feature_size = 427
embedding_size = 10
deep_layers = [1600, 800, 200]
deep_layers_activation = tf.nn.relu
batch_size = 256
learning_rate = 0.001
optimizer_type = "adam"
train_phase = 0
batch_norm_decay = 0.995
verbose = False
random_seed = 201810
l2_reg = 0.0
greater_is_better = True
epoch_number = 1000000


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


def parse_pred_data_func(batch_serialized_example):
    data = {
        "userId": tf.FixedLenFeature([1], tf.float32),
        "songId": tf.FixedLenFeature([1], tf.float32),
        "features": tf.FixedLenFeature([feature_size], tf.float32),
        "label": tf.FixedLenFeature([1], tf.float32)
    }
    data_batch = tf.parse_example(batch_serialized_example, features=data)
    return data_batch


def read_train_data(train_path, num_parallel_reads=8, name=None):
    train_files = get_file_name_list(train_path)
    train = tf.data.TFRecordDataset(train_files, num_parallel_reads=num_parallel_reads) \
        .repeat(2) \
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


def read_pred_data(config, num_parallel_reads=8, name=None):
    dir_file_name_list = []
    index = int(config.index)
    filenames = [f.split()[-1] for f in os.popen('hadoop fs -ls ' + config.input)]
    for f in filenames:
        if 'part' in f and not f.endswith("_COPYING_") and not f.endswith("SUCCESS"):
            dir_file_name_list.append(f)
    pred_data = dir_file_name_list[index * 500:(index + 1) * 500]

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
    dnn0_input_size = feature_size * embedding_size  # N * K
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
    concat_input_size = feature_size + embedding_size + deep_layers[-1]
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
    y_first_order = inputs  # None * N

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
    y_deep = tf.reshape(embeddings, shape=[-1, feature_size * embedding_size])  # None * (N*K)
    for i in range(0, len(deep_layers)):
        y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" % i]), weights["bias_%d" % i])  # None * layer[i] * 1
        if batch_norm:
            y_deep = batch_norm_layer(y_deep, train_phase=train_phase, scope_bn="bn_%d" % i)  # None * layer[i] * 1
        y_deep = deep_layers_activation(y_deep)

    # --- DeepFM ---
    concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
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

    next_train_batch = read_train_data(config.train, num_parallel_reads=8, name="train")
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
                                           train_phase: True})
                s = [k[0] for k in res1[0]]
                l = [k[0] for k in train_batch["label"]]
                auc = roc_auc_score(l, s)
                print("Train--%d--Accuracy:" % i, round(res1[1], 8), "loss:", round(res1[2], 8),
                      "AUC:", round(auc, 8), "concat_bias:", round(res1[3], 8))
            if i % 500 == 0:
                dev_batch = sess.run(next_dev_batch)
                res2 = sess.run([score, accuracy, loss],
                                feed_dict={inputs: dev_batch["features"], labels: dev_batch["label"],
                                           train_phase: True})
                l = [k[0] for k in dev_batch["label"]]
                s = [k[0] for k in res2[0]]
                all_dev_label.extend(l)
                all_dev_score.extend(s)
                auc = roc_auc_score(l, s)
                print("Test--%d--Accuracy:" % i, round(res2[1], 8), "loss:", round(res2[2], 8), "AUC:", round(auc, 8))
            if i % 1000 == 0:
                saver.save(sess, ckpt_dir + '/v2', global_step=i)
                print("concat_projection: %s" % round(res1[3], 8))

        endTime = time.mktime(time.localtime())
        print("Finished, time consumed: %s hours %s minutes %s seconds." % (
            str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
            str((endTime - startTime) % 60)))
        all_dev_auc = roc_auc_score(all_dev_label, all_dev_score)
        print("All Dev AUC = %f" % all_dev_auc)


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
                "input:0": test_batch["features"], "label:0": test_batch["label"], "train_phase:0": True})
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


def prediction1(config):
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
                "input:0": test_batch["features"], "label:0": test_batch["label"], "train_phase:0": True})
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
                "input:0": test_batch["features"], "label:0": test_batch["label"], "train_phase:0": True})
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


def testResult():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        pb_path = os.path.join('/home/ndir/zhangying5/DeepFM/pb/deepfm_v2_2018-10-22.pb')

        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        graph = tf.get_default_graph()

    with tf.Session(graph=graph, config=tf_config) as sess:
        out = graph.get_tensor_by_name("score:0")

        f1 = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12755571, 0.54919595, 0.56761664, 0.0, 0.2630541, 0.768642, 0.21101663,
               0.0, 0.0, 0.40416119, 0.93255687, -2.2637343285714286, -2.050029361428572, 7.065876057142857,
               -1.4820608814285714, 2.677973464285714, 0.8729374550000001, 2.2846453292857145, 3.4023765285714282,
               -2.1961478357142856, -1.8799962714285716, -3.5463740500000003, -6.121168007142857, 2.309055487142857,
               0.4234665357142858, -7.0845753500000015, 5.192194591428572, -3.0711671, 6.1102836371428575,
               0.04951882142857142, 8.359054364285713, 2.0023630128571432, -1.3604826, -6.9316942, 0.8559069071428573,
               -4.7478083, -0.10291444785714286, -1.5925545150000002, -5.257249957142857, -0.34652800485714286,
               7.567918185714285, -3.213643014285714, 6.425833030714287, -1.9739249535714287, 3.191459721428571,
               0.974655432857143, 5.751384591428572, -3.596621428571429, -6.360124725714285, 3.218151685714285,
               5.483406092857143, 3.989399166428571, -8.615902171428571, 5.4228660642857145, -0.14251142142857118,
               -1.1858897114285716, 1.4628895000000004, -2.955393721428572, 5.376218644285714, 1.7414206792857143,
               -11.718255164285715, -4.833487842857143, -0.41791415857142855, -1.1146597921428572, 4.122363621428573,
               -2.6497366928571426, -10.901128557142856, -6.716029030714285, 2.354840435714286, -7.986784857142857,
               -3.0598561, 2.0460023085714285, 2.3243213230714286, -6.9918366, -0.31690174285714284, -7.866464992857142,
               0.1718063442857142, 2.600705272142857, -3.5702486928571426, -4.99434775, -2.7007530021428576,
               3.395213792857143, 1.2431123714285714, -7.594222014285716, -3.0254443, -1.5240608928571429,
               -8.233299535714286, 1.044681422142857, -2.652950680714285, -1.0740051071428571, 3.791753626428571,
               -2.1967446428571433, 1.2547129714285719, 5.81230925, 0.4537013164285715, 3.962038107142858,
               -13.268960935714286, -0.015069578571428719, -0.8859863785714287, -7.9631668357142855, 3.8431059428571426,
               -6.917824628571429, -2.485425492857143, -0.9462782042857143, 5.328227763357143, 0.747613907142857,
               1.8070933714285713, 3.1980730921428577, -6.490684535714287, 1.5781834728571427, -10.366502542857145,
               -4.4046431, -0.55432981, -0.032706775, -0.77871186, -0.70951927, 2.4778256, 0.064140588, 0.2487371,
               3.0682247, -0.64551914, -1.0514137, 0.6091755, 2.1735344, -2.661092, -3.7480419, 0.30258808, 0.72452551,
               0.46085379, 1.0636057, 2.8206365, -2.5693648, -1.7451997, -2.5853732, 0.29744163, -1.6281196, 0.13746373,
               0.45486212, 1.3362899, -2.0830793, 3.5443449, 0.0055646845, 1.8609279, -1.3734, 1.6966004, 1.4434618,
               1.696355, -0.035923079, 1.2107948, 2.6136167, 0.7487306, -2.599365, -3.7200871, -0.34154326, 1.2136865,
               -3.2052267, 0.92274094, -0.57734799, 1.4898925, -0.10172451, -2.9350171, -1.4560213, -0.84829903,
               -1.9212446, 3.1428633, -0.54684633, -3.3652079, 0.19823724, 1.8983718, -1.8263834, 1.2033037, 0.49292591,
               1.4911696, -2.6629765, 2.8476338, -2.0679204, -2.8813517, -2.2133923, -0.21666193, 1.0564702, 0.44056761,
               -1.1408927, 1.1647751, -1.5284793, -1.4872022, 0.32398424, 0.16531679, -0.18983835, -1.182001,
               0.75895208,
               -0.66268331, -1.8532163, -0.84184945, 1.729387, -2.2215066, 2.2002599, -5.3369441, -0.51306975,
               -0.44688317, -1.6483036, -0.20758426, 3.0270579, -2.982964, 1.9531666, -3.4468858, -0.99872297,
               1.1721555,
               4.1139412, -4.2354784, 1.0604538, -1.5414127, 0.18489508, 0.0, 0.0, 0.0, 0.0, 0.14361806, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.57082665, 0.0, 0.0, 0.0, 0.0, 0.06417539, 0.8727112, 0.05440955, 0.0, 0.061477024, 0.0, 0.0,
               0.20342942, 0.060897507, 0.0, 0.0, 0.0, 0.15772384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45048013, 0.0, 0.0, 0.0,
               0.0, 0.10593131, 0.8668207, 0.0, 0.0, 0.08125797, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        f2 = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12755571, 0.44844955, 0.12809238, 0.0, 0.27100068, 0.768642, 0.28140274,
               0.0, 0.0, 0.42927757, 0.121251546, -2.2637343285714286, -2.050029361428572, 7.065876057142857,
               -1.4820608814285714, 2.677973464285714, 0.8729374550000001, 2.2846453292857145, 3.4023765285714282,
               -2.1961478357142856, -1.8799962714285716, -3.5463740500000003, -6.121168007142857, 2.309055487142857,
               0.4234665357142858, -7.0845753500000015, 5.192194591428572, -3.0711671, 6.1102836371428575,
               0.04951882142857142, 8.359054364285713, 2.0023630128571432, -1.3604826, -6.9316942, 0.8559069071428573,
               -4.7478083, -0.10291444785714286, -1.5925545150000002, -5.257249957142857, -0.34652800485714286,
               7.567918185714285, -3.213643014285714, 6.425833030714287, -1.9739249535714287, 3.191459721428571,
               0.974655432857143, 5.751384591428572, -3.596621428571429, -6.360124725714285, 3.218151685714285,
               5.483406092857143, 3.989399166428571, -8.615902171428571, 5.4228660642857145, -0.14251142142857118,
               -1.1858897114285716, 1.4628895000000004, -2.955393721428572, 5.376218644285714, 1.7414206792857143,
               -11.718255164285715, -4.833487842857143, -0.41791415857142855, -1.1146597921428572, 4.122363621428573,
               -2.6497366928571426, -10.901128557142856, -6.716029030714285, 2.354840435714286, -7.986784857142857,
               -3.0598561, 2.0460023085714285, 2.3243213230714286, -6.9918366, -0.31690174285714284, -7.866464992857142,
               0.1718063442857142, 2.600705272142857, -3.5702486928571426, -4.99434775, -2.7007530021428576,
               3.395213792857143, 1.2431123714285714, -7.594222014285716, -3.0254443, -1.5240608928571429,
               -8.233299535714286, 1.044681422142857, -2.652950680714285, -1.0740051071428571, 3.791753626428571,
               -2.1967446428571433, 1.2547129714285719, 5.81230925, 0.4537013164285715, 3.962038107142858,
               -13.268960935714286, -0.015069578571428719, -0.8859863785714287, -7.9631668357142855, 3.8431059428571426,
               -6.917824628571429, -2.485425492857143, -0.9462782042857143, 5.328227763357143, 0.747613907142857,
               1.8070933714285713, 3.1980730921428577, -6.490684535714287, 1.5781834728571427, -10.366502542857145,
               0.024019409, 1.2260107, 0.23766619, -1.3577451, 0.31506193, 0.67530078, 1.123161, -0.012090494,
               -1.5120641, 0.4723511, -0.53307474, -1.3896855, -2.1614656, -0.47122419, -3.9225919, -0.84215772,
               -3.142566, 1.5605049, -1.7205309, 2.5898361, 2.0484552, -1.3716402, -1.3729925, 1.5473272, -1.1679585,
               -0.6122393, 0.62525499, -0.61605418, 3.1688116, 1.3978841, -1.5824311, -0.44110426, -1.5786231,
               0.25269657, 1.4535166, 2.6021013, 0.92807972, 0.42047065, 4.5512328, 5.1058621, 1.0614717, -1.7837809,
               0.43696713, 1.7344804, -3.4901664, -0.34269941, -0.02154251, 1.011248, -0.67438513, -0.47966564,
               -1.6471183, 1.0707887, -1.963323, -0.25260553, -3.5246344, -0.42590207, -0.4647716, -0.46292585,
               -0.90975887, -0.90874487, 3.2876811, 2.0890889, -0.82256925, -2.9197652, -1.4606777, -1.4486542,
               3.3881848, 0.4787868, 0.30773616, -1.1321225, -0.61789882, -2.7260888, -2.519367, -1.2410563, 1.3792073,
               1.2197323, -0.35617828, -2.7176914, -1.840539, 1.4361774, -1.3079352, 0.89450455, 1.6658261, 0.75277907,
               0.21473075, -3.2408721, -1.1775919, -0.61076176, -2.5096695, -1.957526, -0.07166253, 1.1745613,
               -0.84542269, 1.8040799, 1.6000544, -1.9948238, -0.51186401, -1.9654453, 0.45441687, -0.73912919,
               0.18489508, 0.0, 0.0, 0.0, 0.0, 0.14361806, 0.0, 0.0, 0.0, 0.0, 0.0, 0.57082665, 0.0, 0.0, 0.0, 0.0,
               0.06417539, 0.8727112, 0.05440955, 0.0, 0.061477024, 0.0, 0.0, 0.20342942, 0.060897507, 0.0, 0.0, 0.0,
               0.15772384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45048013, 0.0, 0.0, 0.0, 0.0, 0.10593131, 0.8668207, 0.0, 0.0,
               0.08125797, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               1.0]]

        s1 = sess.run(out, feed_dict={"input:0": f1})
        s2 = sess.run(out, feed_dict={"input:0": f2})
        print(s1)
        print(s2)

    f.close()
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', default='v2', required=True)
    parser.add_argument('--mode', help='train or test or pred', required=True)
    parser.add_argument('--gpuid', help='gpu id to use', default='0', required=True)

    parser.add_argument('--index', help='index of index', default="0")  # pred
    parser.add_argument('--input', help='hdfs path of tfrecord', default='v2')  # pred
    parser.add_argument('--output', help='local path of score txt', default='v2')  # pred

    parser.add_argument('--train', help='hdfs path of train tfrecord', default='v2')  # train
    parser.add_argument('--dev', help='hdfs path of dev tfrecord', default='v2')  # train

    parser.add_argument('--test', help='hdfs path of test tfrecord', default='v2')  # test

    config = parser.parse_args()
    return config


if __name__ == '__main__':
    startTime = time.mktime(time.localtime())

    testResult()

    endTime = time.mktime(time.localtime())
    print("Finished, time consumed: %s hours %s minutes %s seconds." % (
        str(int((endTime - startTime) / 3600)), str(int(((endTime - startTime) % 3600) / 60)),
        str((endTime - startTime) % 60)))
