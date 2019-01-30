__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/28
import argparse
import tensorflow as tf
from process_data.census_dataset import train_input_fn, eval_input_fn, build_model_columns
from absl import app
from absl import flags

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
    pred_label = tf.argmax(input=probabilities, axia=1, name="pred_label")
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=probabilities, name="loss")
    acc = tf.metrics.accuracy(labels=labels, predictions=pred_label, name="accuracy")

    if mode == tf.estimator.ModeKey.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKey.EVAL:
        eval_metric_ops = {
            "accuracy": acc,
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKey.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_label)

    else:
        raise KeyError("Unrecognized mode.")


def build_model():
    base_columns, crossed_columns, deep_columns = build_model_columns()
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={"CPU":0},
                                      inter_op_parallelism_threads=1,
                                      intra_op_parallelism_threads=1)
    )

    model_dir = "DNNModel"
    model_params = {
        "feature_columns": deep_columns,
        "num_label": 2
    }
    model = tf.estimator.Estimator(model_fn=dnn_model_fn,
                                   model_dir=model_dir,
                                   config=run_config,
                                   params=model_params
                                   )
    return model


def train_model(model):
    train_epochs = 10
    log_tensors = {
        "loss": "loss",
        "accuracy": "accuracy"
    }
    train_hooks = tf.train.LoggingTensorHook(tensors=log_tensors, every_n_iter=100)
    for n in range(train_epochs // FLAGS.epochs_between_evals):
        model.train(input_fn=train_input_fn, hooks=train_hooks)
        results = model.evaluate(input_fn=eval_input_fn)
        print("Result at epoch %d / %d", ((n+1)*FLAGS.epochs_between_evals), train_epochs)
        print("-" * 60)
        for key in sorted(results):
            print("%s : %s", (key, results[key]))


def eval_model(model):
    results = model.evaluate(input_fn=eval_input_fn)
    for key in sorted(results):
        print("%s : %s", (key, results[key]))


def test_model(model):
    predict_labels = model.predict(input_fn=eval_input_fn)
    print("predict label shape: %s", predict_labels.shape)


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




def main(argv):
    model = build_model()
    train_model(model)
    eval_model(model)
    test_model(model)
    save_model(model, FLAGS)
    model = load_model(FLAGS)
    eval_model(model)




if __name__ == "__main__":
    define_flags()
    app.run(main)