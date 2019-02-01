__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/31
import argparse


class TFBaseModel(object):
    def __init__(self):
        self.in_feature = None
        self.in_label = None
        self.loss = None
        self.train_op = None
        self.eval_metric_ops = {}
        self.prediction = None

    def setup_input(self, feature, label):
        self.in_feature = feature
        self.in_label = label

    def setup_train(self, loss, train_op):
        self.loss = loss
        self.train_op = train_op

    def setup_eval(self, loss, eval_metric_ops):
        self.loss = loss
        self.eval_metric_ops = eval_metric_ops

    def setup_predict(self, prediction):
        self.prediction = prediction

    def train(self, feature, label, sess):
        feed_dict={
            self.in_feature: feature,
            self.in_label: label
        }
        _ = sess.run(self.train_op, feed_dict=feed_dict)

    def eval(self, feature, label, sess):
        """

        :param feature: sample feature
        :param label: sample label
        :param sess: session
        :return: res = {key:val}
        """
        kvs = self.eval_metric_ops.items()
        keys = [v[0] for v in kvs]
        vals = [v[1] for v in kvs]
        feed_dict = {
            self.in_feature: feature,
            self.in_label: label
        }
        eval_res = sess.run(vals, feed_dict=feed_dict)
        res = {}
        for i, k in enumerate(keys):
            res[k] = eval_res[i]
        return res

    def predict(self, feature, sess):
        feed_dict = {
            self.in_feature: feature
        }
        pred_res = sess.run([self.prediction], feed_dict=feed_dict)
        return pred_res[0]


def main(config):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)

