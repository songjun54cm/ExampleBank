__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/12/15
import argparse
import tensorflow as tf
from iris.iris_script import load_data

def main(config):
    model_file = "../data/iris/output/frozen/iris.pb"
    with tf.gfile.GFile(model_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # tensorflow adds "import/" prefix to all tensors when imports graph definition, ex: "import/input:0"
        # so here explicitly tell tensorflow to use empty string -> name=""
        tf.import_graph_def(graph_def, name="")
        # just print all operations for debug
        print(tf.get_default_graph().get_operations())

        def_g = tf.get_default_graph()
        input_fea = def_g.get_tensor_by_name("input_fea:0")
        input_label = def_g.get_tensor_by_name("input_label:0")
        pred_probs = def_g.get_tensor_by_name("pred_probs:0")
        pred_class = def_g.get_tensor_by_name("pred_class:0")
        pred_accuracy = def_g.get_tensor_by_name("pred_accuracy:0")

        train_fea, train_label, eval_fea, eval_label = load_data()
        with tf.Session() as sess:
            pred_acc, pred_cls, pred_prbs = sess.run([
                pred_accuracy,
                pred_class,
                pred_probs
            ],feed_dict={
                input_fea: eval_fea,
                input_label: eval_label
            })

        print("accuracy; %f" % pred_acc)
        print("predict class %s" % str(pred_class))
        print("predict probs %s" % str(pred_probs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)