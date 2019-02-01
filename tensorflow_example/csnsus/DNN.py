__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/31
import argparse
import tensorflow as tf
from TFBaseModel import TFBaseModel
from absl import app
from absl import flags
from process_data.census_dataset import train_input_iter, eval_input_iter, build_model_columns


FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_integer(name="iters_between_evals",
                         default=10,
                         help="number of iters between evals.",
                         lower_bound=1,
                         upper_bound=10000)
    flags.DEFINE_integer(name="epochs_between_evals",
                         default=1,
                         help="number of epochs between evals.",
                         lower_bound=1,
                         upper_bound=100)
    flags.DEFINE_string(name="export_dir",
                        default="dnn_model",
                        help="model export dir.")


class DNNModel(TFBaseModel):
    def __init__(self, config):
        """

        :param config:{
            "feature_dim": x,
            "num_label": x
        }
        """
        super(DNNModel, self).__init__()
        self.create_model(config)

    def create_model(self, config):
        input_feature = tf.placeholder(tf.float32, [None, config["feature_dim"]], name="input_feature")
        input_label = tf.placeholder(tf.int32, [None], name="input_label")
        self.setup_input(feature=input_feature, label=input_label)

        full1 = tf.layers.dense(inputs=input_feature, units=20, activation=tf.nn.relu, name="full1")
        full2 = tf.layers.dense(inputs=full1, units=10, activation=tf.nn.relu, name="full2")
        full3 = tf.layers.dense(inputs=full2, units=10, activation=tf.nn.relu, name="full3")
        out_layer = tf.layers.dense(inputs=full3, units=config["num_label"], activation=tf.nn.sigmoid, name="out")

        probabilities = tf.nn.softmax(out_layer, name="probs")
        pred_label = tf.argmax(input=probabilities, axis=1, name="pred_label")
        loss = tf.losses.sparse_softmax_cross_entropy(labels=input_label, logits=probabilities)
        acc, acc_op = tf.metrics.accuracy(labels=input_label, predictions=pred_label, name="accuracy")
        # for train
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss)
        eval_metric_ops = {
            "accuracy": acc_op
        }

        self.setup_train(loss=loss, train_op=train_op)
        self.setup_eval(loss=loss, eval_metric_ops=eval_metric_ops)
        self.setup_predict(prediction=pred_label)


def eval_model(model, input_fea, input_label, eval_iter, sess):
    total_result = {}
    total_sample = 0
    sess.run(eval_iter.initializer)
    try:
        while True:
            fea_v, label_v = sess.run([input_fea, input_label])
            result = model.eval(fea_v, label_v, sess)
            num_sample = fea_v.shape[0]
            total_sample += num_sample
            for key in sorted(result):
                total_result[key] = total_result.get(key, 0) + (result[key] * num_sample)
    except tf.errors.OutOfRangeError:
        for key in sorted(total_result):
            total_result[key] = total_result[key] / total_sample
    return total_result


def train_model(model, input_fea, input_label, train_iter, eval_in_fea, eval_in_label, eval_iter, sess):
    train_epochs = 20
    for n in range(train_epochs):
        sess.run(train_iter.initializer)
        iter_n = 0
        try:
            print("train epoch %d" % n)
            while True:
                iter_n += 1
                fea_v, label_v = sess.run([input_fea, input_label])
                # print("fea_v shape: %s" % str(fea_v.shape))
                model.train(fea_v, label_v, sess)
                if iter_n % FLAGS.iters_between_evals == 0:
                    print("eval epoch %d" % n)
                    result = eval_model(model, eval_in_fea, eval_in_label, eval_iter, sess)
                    print("Result at iter %d" % iter_n)
                    for key in sorted(result):
                        print("%s : %s" % (key, result[key]))
        except tf.errors.OutOfRangeError:
            print("out of range. eval epoch %d"% n)
            result = eval_model(model, eval_in_fea, eval_in_label, eval_iter, sess)
            print("Result at iter %d, epoch %d" % (iter_n, n))
            for key in sorted(result):
                print("%s : %s" % (key, result[key]))
                print("=="*20)
    return sess

def then_test_model(model, test_dataset, sess):
    pass


def save_model(model, flag_obj):
    pass


def load_model(flag_obj):
    pass


def main(argv):
    config = {
        "feature_dim": 1043,
        "num_label": 2
    }
    model = DNNModel(config)
    _, _, deep_columns = build_model_columns()

    train_iter = train_input_iter()
    tr_fea, train_in_label = train_iter.get_next()
    train_in_fea = tf.feature_column.input_layer(tr_fea, deep_columns)
    eval_iter = eval_input_iter()
    eval_fea, eval_in_label = eval_iter.get_next()
    eval_in_fea = tf.feature_column.input_layer(eval_fea, deep_columns)
    with tf.Session() as sess:
        init_op = tf.group(tf.tables_initializer(),
                           tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        train_model(model, train_in_fea, train_in_label, train_iter, eval_in_fea, eval_in_label, eval_iter, sess)
        eval_model(model, eval_in_fea, eval_in_label, eval_iter, sess)


if __name__ == "__main__":
    define_flags()
    app.run(main)
