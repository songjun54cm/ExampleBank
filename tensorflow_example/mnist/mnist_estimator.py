__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/12/12
import argparse
import argparse
import numpy as np
import tensorflow as tf
import random
import os
import time


def load_data():
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
    return train_data, train_labels, eval_data, eval_labels


def my_model_fn(features, labels, mode, params):
    input_fea = tf.feature_column.input_layer(features, params["feature_columns"])
    # input_label = tf.placeholder(tf.int64, [None], name="input_label")
    # input_fea = tf.placeholder(tf.float32, [None,28,28,1], name="input_fea")
    conv1 = tf.layers.conv2d(
        inputs=input_fea,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    logits = tf.layers.dense(inputs=dropout, units=10)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')
    pred_class = tf.argmax(input=logits, axis=1, name='pred_class')

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "class_ids": pred_class,
            "probabilities": probabilities,
            "logits": logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKey.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=pred_class)
    }

    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_input_fn(features, labels, epoch_num, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)) \
        .shuffle(10*batch_size) \
        .repeat(epoch_num) \
        .batch(batch_size)
    return dataset


def eval_input_fn(feature, labels, batch_size):
    features = dict(feature)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset


def main(config):
    train_data, train_labels, eval_data, eval_labels = load_data()

    my_feature_columns = list()
    my_feature_columns.append(tf.feature_column.numeric_column(key="fea",
                                                               shape=train_data.shape[1:]))
    train_x = {
        "fea": train_data
    }
    eval_x = {
        "fea": eval_data
    }

    # save check point into ckpt_dir
    ckpt_dir = "../data/output/mnist/estimator"
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # set checkpoint schedule
    ckpt_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20*60, # save checkpoint every 20 minutes
        keep_checkpoint_max=10, # Retain the 10 most recent checkpoint
    )

    classifier = tf.estimator.Estimator(model_fn=my_model_fn,
                                        params={
                                            "fea":my_feature_columns
                                        },
                                        model_dir="../data/output/mnist/estimator",
                                        config=ckpt_config)

    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_labels, 10, 100))

    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(eval_x, eval_labels, 10)
    )
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(eval_x,batch_size=10)
    )

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, eval_labels):
        class_id = pred_dict['class'][0]
        probability = pred_dict['prob'][class_id]
        print(template.format(class_id,100 * probability, expec))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)