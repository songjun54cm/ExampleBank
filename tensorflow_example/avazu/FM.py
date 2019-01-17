__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/14
import argparse
import tensorflow as tf
import pickle
import os
import numpy as np
from settings import DATA_DIR
import pandas as pd
from data_process.utilities import train_batch_sparse_data_generate, ttest_sparse_data_generate
import json


class FM(object):
    def __init__(self, config):
        self.k = config["k"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.reg_l1 = config["reg_l1"]
        self.reg_l2 = config["reg_l2"]
        self.p = config["feature_length"]
        self.create_model()

    def create_model(self):
        self.X = tf.sparse_placeholder('float32', [None, self.p])
        self.y = tf.placeholder('int64', [None,])
        self.keep_prob = tf.placeholder('float32')

        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.sparse_tensor_dense_matmul(self.X, w1), b)

        with tf.variable_scope('interaction_layer'):
            v = tf.get_variable('v', shape=[self.p, self.k],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.interaction_terms = tf.multiply(0.5,
                                                 tf.reduce_mean(
                                                     tf.subtract(
                                                         tf.pow(tf.sparse_tensor_dense_matmul(self.X, v), 2),
                                                         tf.sparse_tensor_dense_matmul(self.X, tf.pow(v, 2))),
                                                     1, keep_dims=True))
        # shape of [None, 2]
        self.y_out = tf.add(self.linear_terms, self.interaction_terms)
        self.y_out_prob = tf.nn.softmax(self.y_out)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

        # accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out,1), tf.int64), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # add summary to accuracy
        tf.summary.scalar('accuracy', self.accuracy)

        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # define optimizer
        optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


def train_model(sess, model, fields_dict, epochs=10, print_every=50):
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)

    for e in range(epochs):
        num_samples = 0
        losses = []
        # get training data
        train_data = pd.read_csv('G://Datasets//avazuCTR//train.csv', chunksize=model.batch_size)
        for data in train_data:
            actual_batch_size = len(data)
            indexes, labels = train_batch_sparse_data_generate(data, fields_dict)
            batch_indexes = np.array(indexes, dtype=np.int64)
            batch_shape = np.array([actual_batch_size, feature_length], dtype=np.int64)
            batch_values = np.ones(len(batch_indexes), dtype=np.float32)
            batch_y = labels

            feed_dict = {model.X: (batch_indexes, batch_values, batch_shape),
                         model.y: batch_y,
                         model.keep_prob: 1.0}
            loss, accuracy, summary, global_step, _ = sess.run([model.loss, model.accuracy,
                                                                merged, model.global_step,
                                                                model.train_op], feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            num_samples += actual_batch_size
            # Record summaries and train.csv-set accuracy
            train_writer.add_summary(summary, global_step=global_step)
            if global_step % print_every == 0:
                print("Iteration: %d, loss: %f, accuracy: %f" % (global_step, loss, accuracy))

        total_loss = np.sum(losses) / num_samples
        print("Epoch: %d, Overall loss: %.3f" % (e, total_loss))


def then_test_model(sess, model, fields_dict, print_every=50):
    """testing model"""
    # get testing data, iterable
    test_data = pd.read_csv('G://Datasets//avazuCTR//test.csv', chunksize=model.batch_size)
    all_ids = []
    all_clicks = []
    ibatch = 0
    for data in test_data:
        indexes, ids = ttest_sparse_data_generate(data, fields_dict)
        actual_batch_size = len(data)
        batch_indexes = np.array(indexes, dtype=np.int64)
        batch_shape = np.array([actual_batch_size, feature_length], dtype=np.int64)
        batch_values = np.ones(len(batch_indexes), dtype=np.float32)
        # create a feed dictionary for this15162 batch
        feed_dict = {model.X: (batch_indexes, batch_values, batch_shape),
                     model.keep_prob:1.0}
        # shape of [None,2]
        y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
        batch_clicks = y_out_prob[0][:,-1]
        all_ids.extend(ids)
        all_clicks.extend(batch_clicks)
        ibatch += 1
        if ibatch % print_every == 0:
            print("Iteration %d has finished" % ibatch)


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
        "reg_l1": 2e-2,
        "reg_l2": 0,
        "k": 40,
        "feature_length": test_array_len
    }
    print(json.dumps(config))
    feature_length = test_array_len
    model = FM(config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("start training...")
        train_model(sess, model, fields_dict, epochs=20, print_every=500)
        then_test_model(sess, model, fields_dict)
