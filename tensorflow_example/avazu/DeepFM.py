__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/17
import argparse
import tensorflow as tf


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
            
