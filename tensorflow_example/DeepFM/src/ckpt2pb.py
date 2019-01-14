#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


def main(config):
    g = tf.get_default_graph()

    with tf.Session(graph=g) as sess:
        # Load .ckpt file
        modle_path = os.path.join('../checkpoint', config.model)
        ckpt_path = modle_path + "/" + config.ckpt_v
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)

        # Save as .pb file
        graph_def = g.as_graph_def()

        # fix batch norm nodes
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']

        if int(config.mode) == 1:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['score']
            )
        elif int(config.mode) == 2:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['score1', 'score2'],
            )

        pb_path = os.path.join('../pb', config.model)
        with tf.gfile.GFile(pb_path + ".pb", 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)
    return


def testPB(config):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        pb_path = os.path.join('../pb', config.model)

        with open(pb_path + ".pb", "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        graph = tf.get_default_graph()
        # for op in graph.get_operations():
        #     print(op.name, op.values())
        f = open("../sh/node", "w")
        for node in output_graph_def.node:
            f.write(str(node) + "\n")
        f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model name', required=True)
    parser.add_argument('--ckpt_v', help='checkpoint version', required=True)
    parser.add_argument('--gpuid', help='gpuid', required=True)
    parser.add_argument('--mode', help='mode', required=True)
    config = parser.parse_args()
    return config


if __name__ == '__main__':
    config = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    main(config)

    testPB(config)
