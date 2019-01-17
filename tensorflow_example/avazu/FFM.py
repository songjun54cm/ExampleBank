__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/15
import argparse
import tensorflow as tf
import pandas as pd
import os
from settings import DATA_DIR
from data_process.utilities import ffm_batch_data_generate
import pickle
import numpy as np


class FFM(object):
    """
    Field-aware Factorization Machine
    """
    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = config['k']
        # num of fields
        self.f = config['f']
        # num of features
        self.p = config["feature_length"]
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']
        self.feature2field = config['feature2field']
        self.create_model()

    def create_model(self):
        self.X = tf.placeholder("float32", [self.batch_size, self.p])
        self.y = tf.placeholder("int64", [None,])
        self.keep_prob = tf.placeholder("float32")

        """
        forward propagation
        :return: labels for each sample
        """
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[2],
                                initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.matmul(self.X, w1), b)

        with tf.variable_scope('field_aware_interaction_layer'):
            v = tf.get_variable('v', shape=[self.p, self.f, self.k], dtype='float32',
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.field_aware_interaction_terms = tf.constant(0, dtype='float32')
            # build dict to find f, key of feature, value of field
            for i in range(self.p):
                for j in range(i+1, self.p):
                    self.field_aware_interaction_terms += tf.multiply(
                        tf.reduce_sum(tf.multiply(v[i,self.feature2field[i]], v[j,self.feature2field[j]])),
                        tf.multiply(self.X[:,i],self.X[:,j])
                    )
        # shape of [None, 2]
        self.y_out = tf.add(self.linear_terms, self.field_aware_interaction_terms)
        self.y_out_prob = tf.nn.softmax(self.y_out)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out, 1), tf.int64), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdagardDAOptimizer(self.lr, global_step=self.global_step)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


def train_model(sess, model, fields_dict, feature_length, epochs=10, print_every=500):
    """training model"""
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)
    for e in range(epochs):
        num_samples = 0
        losses = []
        train_data = pd.read_csv('G://Datasets//avazuCTR//train.csv', chunksize=model.batch_size)
        for data in train_data:
            actual_batch_size = len(data)
            batch_x, batch_y = ffm_batch_data_generate(data, fields_dict, feature_length)
            feed_dict = {
                model.X: batch_x,
                model.y: batch_y,
                model.keep_prob: 1.0
            }
            loss, accuracy, summary, global_step, _ = sess.run([model.loss, model.accuracy, merged,
                                                                model.global_step, model.train_op], feed_dict=feed_dict)
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


def test_model(sess, model, fields_dict, feature_length, epochs=10, print_every=500):
    """testing model"""


if __name__ == "__main__":
    fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
              'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
              'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
              'device_conn_type','click']
    fields_dict = {}
    for field in fields:
        with open(os.path.join(DATA_DIR, "dicts", "%s.pkl"%field), "rb") as f:
            fields_dict[field] = pickle.load(f)

    with open(os.path.join(DATA_DIR, "dicts", "feature2field.pkl"), "rb") as f:
        feature2field = pickle.load(f)

    train_array_len = max(fields_dict["click"].values()) + 1
    test_array_len = train_array_len - 2
    config = {
        "lr": 0.01,
        "batch_size": 512,
        "reg_l1": 2e-3,
        "reg_l2": 0,
        "k": 4,
        "f": len(fields) - 1,
        "feature2field": feature2field
    }
    feature_length = test_array_len
    model = FFM(config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_model(sess, model, fields_dict, feature_length)
        test_model(sess, model, fields_dict, feature_length)
