__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/25
import argparse
import tensorflow as tf


def lr_model_fn(features, labels, mode, params):
    input_fea = tf.feature_column.input_layer(features, params["feature_columns"])



def main(config):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)
