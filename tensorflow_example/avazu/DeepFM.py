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


