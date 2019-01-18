__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/17
import argparse
import tensorflow as tf
from settings import DATA_DIR
import pandas as pd
import numpy as np
import pickle
import os
import json
from data_process.utilities import deepfm_batch_data_generate

class DeepFM(object):
    def __init__(self, config):
        self.k = config["k"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.reg_l1 = config["reg_l1"]
        self.reg_l2 = config["reg_l2"]
        self.p = config["feature_length"]
        self.field_cnt = config["field_cnt"]
        self.create()

    def create(self):
        self.X = tf.placeholder("float32", [None, self.p])
        self.y = tf.placeholder("int64", [None,])
        self.feature_inds = tf.placeholder("int64", [None, self.field_cnt])
        self.keep_prob = tf.placeholder("float32")

        v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01), dtype="float32")

        # Factorization Machine
        with tf.variable_scope("FM"):
            b = tf.get_variable("bias", shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable("w1", shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.matmul(self.X, w1), b)

            # shape of [None, 1]
            self.interaction_terms = tf.multiply(0.5,
                                                 tf.reduce_mean(
                                                     tf.subtract(
                                                         tf.pow(tf.matmul(self.X, v), 2),
                                                         tf.matmul(self.X, tf.pow(v,2))
                                                     ), 1, keep_dims=True))
            # shape of [None, 2]
            self.y_fm = tf.add(self.linear_terms, self.interaction_terms)

        # three-hidden-layer NN
        with tf.variable_scope("DNN", reuse=False):
            # embedding layer
            y_embedding_input = tf.reshape(tf.gather(v, self.feature_inds), [-1, self.field_cnt*self.k])
            # first hidden layer
            w1 = tf.get_variable("w1_dnn", shape=[self.field_cnt*self.k, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddve=1e-2))
            b1 = tf.get_variable("b1_dnn", shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l1 = tf.nn.relu(tf.matml(y_embedding_input, w1) + b1)
            # second hidden layer
            w2 = tf.get_variable("w2_dnn", shape=[200, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))
            b2 = tf.get_variable("b2_dnn", shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l2 = tf.nn.relu(tf.matmul(y_hidden_l1, w2) + b2)
            # third hidden layer
            w3 = tf.get_variable("w3_dnn", shape=[200, 200],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))
            b3 = tf.get_variable("b3_dnn", shape=[200],
                                 initializer=tf.constant_initializer(0.001))
            y_hidden_l3 = tf.nn.relu(tf.matmul(y_hidden_l2, w3) + b3)
            # output layer
            w_out = tf.get_variable("w_out", shape=[200, 2],
                                    initializer=tf.truncated_normal_initializer(mean=0, stddev=1e-2))
            b_out = tf.get_variable("b_out", shape=[2],
                                    initializer=tf.constant_initializer(0.001))
            self.y_dnn = tf.nn.relu(tf.matmul(y_hidden_l3, w_out) + b_out)

        self.y_out = tf.add(self.y_fm, self.y_dnn)
        self.y_out_prob = tf.nn.softmax(self.y_out)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar("loss", self.loss)

        self.correct_prediction = tf.equal(tf.cast(tf.argmax(model.y_out, 1), tf.int64), model.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", self.accuracy)

        self.global_step = tf.Variable(0, trainable=False)

        optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


def train_model(sess, model, fields_dict, feature_length, epochs=10, print_every=500):
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)

    for e in range(epochs):
        num_samples = 0
        losses = []
        # get training data
        train_data = pd.read_csv('G://Datasets//avazuCTR//train.csv', chunksize=model.batch_size)
        for data in train_data:
            actual_batch_size = len(data)
            batch_x, batch_y, batch_idx = deepfm_batch_data_generate(data, fields_dict, feature_length)
            feed_dict = {model.X: batch_x,
                         model.y: batch_y,
                         model.feature_inds: batch_idx,
                         model.keep_prob:1}
            loss, accuracy, summary, global_step, _ = sess.run([model.loss, model.accuracy,
                                                                merged, model.global_step,
                                                                model.train_op], feed_dict=feed_dict)
            # aggregate performance stats
            losses.append(loss*actual_batch_size)

            num_samples += actual_batch_size
            # Record summaries and train.csv-set accuracy
            train_writer.add_summary(summary, global_step=global_step)
            # print training loss and accuracy
            if global_step % print_every == 0:
                print("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                             .format(global_step, loss, accuracy))
                # saver.save(sess, "checkpoints/model", global_step=global_step)

        # print loss of one epoch
        total_loss = np.sum(losses)/num_samples
        print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss, e+1))


def valid_model(sess, model, field_dict, feature_length, print_every=50):
    merged = tf.summary.merge_all()
    num_samples = 0
    num_corrects = 0
    losses = []
    test_writer = tf.summary.FileWriter('test_logs', sess.graph)
    # get training data
    valid_data = pd.read_csv('G://Datasets//avazuCTR//train.csv', chunksize=model.batch_size)
    valid_step = 1
    for data in valid_data:
        actual_batch_size = len(data)
        batch_x, batch_y, batch_idx = deepfm_batch_data_generate(data, fields_dict, feature_length)
        feed_dict = {model.X: batch_x,
                     model.y: batch_y,
                     model.feature_inds: batch_idx,
                     model.keep_prob:1}
        loss, accuracy, correct, summary = sess.run([model.loss, model.accuracy,
                                                      model.correct_prediction,merged],
                                                     feed_dict=feed_dict)
        # aggregate performance stats
        losses.append(loss*actual_batch_size)
        num_corrects += correct
        num_samples += actual_batch_size
        # Record summaries and train.csv-set accuracy
        test_writer.add_summary(summary, global_step=valid_step)
        # print training loss and accuracy
        if valid_step % print_every == 0:
            print("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                  .format(valid_step, loss, accuracy))
            # saver.save(sess, "checkpoints/model", global_step=global_step)

    # print loss of one epoch
    total_correct = num_corrects / num_samples
    total_loss = np.sum(losses)/num_samples
    print("Overall test loss = {0:.3g} and accuracy of {1:.3g}" \
          .format(total_loss,total_correct))


def then_test_model(sess, model, fields_dict, feature_length, print_every=50):
    # get testing data, iterable
    test_data = pd.read_csv('G://Datasets//avazuCTR//test.csv', chunksize=model.batch_size)
    test_step = 1
    # batch_size data
    for data in test_data:
        actual_batch_size = len(data)
        batch_x, batch_y, batch_idx = deepfm_batch_data_generate(data, fields_dict, feature_length)
        # create a feed dictionary for this batch
        feed_dict = {model.X: batch_x, model.keep_prob:1, model.feature_inds:batch_idx}
        # shape of [None,2]
        y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
        # write to csv files
        data['click'] = y_out_prob[0][:,-1]
        # if test_step == 1:
        #     data[['id','click']].to_csv('Deep_FM_FTRL_v1.csv', mode='a', index=False, header=True)
        # else:
        #     data[['id','click']].to_csv('Deep_FM_FTRL_v1.csv', mode='a', index=False, header=False)

        test_step += 1
        if test_step % 50 == 0:
            print("Iteration {0} has finished".format(test_step))


if __name__ == "__main__":
    fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
              'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
              'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
              'device_conn_type','click']

    fields_dict = {}
    for field in fields:
        with open(os.path.join(DATA_DIR, "dicts", "%s.pkl"%field), "rb") as f:
            fields_dict[field] = pickle.load(f)

    train_array_len = max(fields_dict["click"].values()) + 1
    test_array_len = train_array_len - 2
    config = {
        "lr": 0.01,
        "batch_size": 512,
        "reg_l1": 2e-3,
        "reg_l2": 0,
        "k": 40,
        "feature_length": test_array_len,
        "field_cnt": 21
    }
    print(json.dumps(config))
    model = DeepFM(config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("start training...")
        train_model(sess, model, fields_dict, test_array_len, epochs=20, print_every=500)
        valid_model(sess, model, fields_dict, test_array_len)
        then_test_model(sess, model, fields_dict, test_array_len)
