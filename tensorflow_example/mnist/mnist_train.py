"""
Author: songjun
Date: 2018/04/12
Description:
Usage:
"""
import argparse
import numpy as np
import tensorflow as tf
import random
import os


def main(config):
    # Input Layer
    label_input_layer = tf.placeholder(tf.int64, [None], name='input_label')
    fea_input_layer = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_feature')

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=fea_input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    logits = tf.layers.dense(inputs=dropout, units=10)

    probabilities = tf.nn.softmax(logits, name='softmax_tensor')
    pred_class = tf.argmax(input=logits, axis=1, name='pred_class')
    pred_accuracy = tf.reduce_mean(tf.to_float(tf.equal(pred_class, label_input_layer)), name='pred_accuracy')
    # pred_accuracy = tf.metrics.accuracy(labels=label_input_layer, predictions=pred_class, name='pred_accuracy')

    # Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=label_input_layer, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)


    print('loading data.')
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    print('train data shape %s' % str(train_data.shape))
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    print('train label shape %s' % str(train_labels.shape))
    eval_data = mnist.test.images  # Returns np.array
    eval_data = np.reshape(eval_data, (eval_data.shape[0], 28, 28, 1))
    print('eval data shape %s' % str(eval_data.shape))
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    print('eval labels shape %s' % str(eval_labels.shape))


    batch_size = 1000
    epoch_num = 100

    save_folder = os.path.join('output', config['mode'])
    saved_model_dir = os.path.join(save_folder, 'saved_model')
    saver_dir = os.path.join(save_folder, 'saver')
    if not os.path.exists(save_folder):
        print('make dir %s' % save_folder)
        os.makedirs(save_folder)
        os.makedirs(saved_model_dir)
        os.makedirs(saver_dir)

    saver = tf.train.Saver()


    # builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])

    idx_list = list(range(train_data.shape[0]))
    sample_num = len(idx_list)
    for epoch_i in range(epoch_num+1):
        for _ in range(0, sample_num, batch_size):
            batch_idx = random.sample(idx_list, batch_size)
            batch_fea = train_data[batch_idx]
            batch_fea = np.reshape(batch_fea, (batch_fea.shape[0], 28, 28, 1))
            batch_label = train_labels[batch_idx]
            _, batch_loss = sess.run((train_op, loss), feed_dict={fea_input_layer: batch_fea, label_input_layer:batch_label})
            print('batch loss: %f' % batch_loss)

        acc = sess.run(pred_accuracy, feed_dict={fea_input_layer: eval_data, label_input_layer: eval_labels})
        # print(acc)
        print('epoch %d, evaluate accuracy: %f' % (epoch_i, acc))
        if epoch_i%10 == 0:
            saver_folder = os.path.join(saver_dir, 'saver_epoch_%d' % epoch_i)
            save_path = os.path.join(saver_folder, 'model')
            saver.save(sess, save_path)
            print('saver save model into %s' % save_path)

            export_dir = os.path.join(saved_model_dir, 'epoch_%d' % epoch_i)
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], clear_device=True)
            model_path = builder.save()
            print('SavedModel save model into %s' % model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    parser.add_argument('-m', '--mode', dest='mode', type=str, default='cpu', help='cpu/gpu')
    args = parser.parse_args()
    config = vars(args)
    if config['mode'] == 'gpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    elif config['mode'] == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main(config)

