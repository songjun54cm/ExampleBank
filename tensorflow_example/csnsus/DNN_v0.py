__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/28
import argparse
import tensorflow as tf
from process_data.census_dataset import train_input_fn, eval_input_fn, build_model_columns
from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_integer(name="epochs_between_evals",
                         default=1,
                         help="number of epochs between evals.",
                         lower_bound=0,
                         upper_bound=100)
    flags.DEFINE_string(name="export_dir",
                        default="dnn_model",
                        help="model export dir.")


def dnn_model_fn(features, labels, mode, params):
    """

    :param features:
    :param labels:
    :param mode:
    :param params: {
        "feature_columns": [],
        "num_label": x
    }
    :return:
    """
    input_fea = tf.feature_column.input_layer(features, params["feature_columns"])
    full1 = tf.layers.dense(inputs=input_fea, units=20, activation=tf.nn.relu, name="full1")
    full2 = tf.layers.dense(inputs=full1, units=10, activation=tf.nn.relu, name="full2")
    full3 = tf.layers.dense(inputs=full2, units=10, activation=tf.nn.relu, name="full3")
    out_layer = tf.layers.dense(inputs=full3, units=params["num_label"], activation=tf.nn.sigmoid, name="out")

    probabilities = tf.nn.softmax(out_layer, name="probs")
    pred_label = tf.argmax(input=probabilities, axis=1, name="pred_label")
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=probabilities)
    acc = tf.metrics.accuracy(labels=labels, predictions=pred_label, name="accuracy")

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": acc,
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_label)

    else:
        raise KeyError("Unrecognized mode.")


def build_model(warm_start_from=None):
    base_columns, crossed_columns, deep_columns = build_model_columns()
    run_config = None
    # run_config = tf.estimator.RunConfig().replace(
    #     session_config=tf.ConfigProto(device_count={"CPU":0},
    #                                   inter_op_parallelism_threads=2,
    #                                   intra_op_parallelism_threads=2))

    model_dir = "DNNModel"
    model_params = {
        "feature_columns": deep_columns,
        "num_label": 2
    }
    model = tf.estimator.Estimator(model_fn=dnn_model_fn,
                                   model_dir=model_dir,
                                   config=run_config,
                                   params=model_params,
                                   warm_start_from=warm_start_from)
    return model


def train_model(model):
    train_epochs = 2
    log_tensors = {
        "accuracy": "accuracy"
    }
    train_hooks = [tf.train.LoggingTensorHook(tensors=log_tensors, every_n_iter=100)]
    for n in range(train_epochs // FLAGS.epochs_between_evals):
        model.train(input_fn=train_input_fn, hooks=None)
        results = model.evaluate(input_fn=eval_input_fn)
        print("Result at epoch %d / %d" % (((n+1)*FLAGS.epochs_between_evals), train_epochs))
        print("-" * 60)
        for key in sorted(results):
            print("%s : %s" % (key, results[key]))


def eval_model(model):
    results = model.evaluate(input_fn=eval_input_fn)
    for key in sorted(results):
        print("%s : %s" % (key, results[key]))


def then_test_model(model):
    predict_labels = model.predict(input_fn=eval_input_fn)
    print("predict label shape: %s" % np.ndarray(list(predict_labels)).shape)


def save_model(model, flag_obj):
    base_columns, crossed_columns, deep_columns = build_model_columns()
    feature_spec = tf.feature_column.make_parse_example_spec(deep_columns)
    example_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    )
    model.export_savemodel(export_dir_base=flag_obj.export_dir,
                           serving_input_receiver_fn=example_input_fn,
                           strip_default_attrs=True)


def load_model(flag_obj):
    model = build_model(flag_obj.export_dir)
    return model


def main(argv):
    print("build model...")
    model = build_model()
    print("train model...")
    train_model(model)
    print("op names")
    print(tf.get_default_graph().as_graph_def())
    for op in tf.get_default_graph().get_operations():
        print(str(op.name))
    print("variable names")
    print(model.get_variable_names())
    print("eval model...")
    eval_model(model)
    print("test model...")
    then_test_model(model)
    print("save model...")
    save_model(model, FLAGS)
    print("load model...")
    model = load_model(FLAGS)
    print("eval model...")
    eval_model(model)


if __name__ == "__main__":
    define_flags()
    app.run(main)
