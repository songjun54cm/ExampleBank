__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/12/13
import argparse
import os
import random
import numpy as np
import tensorflow as tf
import _pickle as pkl


def load_data():
    data_path = "../data/iris/data.pkl"
    if os.path.exists(data_path):
        print("data exists, load data.")
        data = pkl.load(open(data_path, "rb"))
        train_fea = data["train_fea"]
        train_label = data["train_label"]
        eval_fea = data["eval_fea"]
        eval_label = data["eval_label"]
    else:
        print("data not exists, download data.")
        from sklearn.datasets import load_iris
        x, y = load_iris(return_X_y=True)
        samples = [(a,b) for (a,b) in zip(x,y)]
        random.shuffle(samples)
        train_samples = samples[:int(len(samples)*0.8)]
        eval_samples = samples[int(len(samples)*0.8):]
        train_fea = np.asarray([x[0] for x in train_samples])
        train_label = np.asarray([x[1] for x in train_samples])
        eval_fea = np.asarray([x[0] for x in eval_samples])
        eval_label = np.asarray([x[1] for x in eval_samples])
        data = {
            "train_fea": train_fea,
            "train_label": train_label,
            "eval_fea": eval_fea,
            "eval_label": eval_label
        }
        pkl.dump(data, open(data_path, "wb"))
    print('train data shape %s' % str(train_fea.shape))
    print('train label shape %s' % str(train_label.shape))
    print('eval data shape %s' % str(eval_fea.shape))
    print('eval labels shape %s' % str(eval_label.shape))
    return train_fea, train_label.astype(np.int64), eval_fea, eval_label.astype(np.int64)


def main(config):
    in_fea = tf.placeholder(tf.float32, [None, 4], name="input_fea")
    in_label = tf.placeholder(tf.int64, [None], name="input_label")

    full1 = tf.layers.dense(
        inputs=in_fea,
        units=10,
        activation=tf.nn.tanh,
        name="full1"
    )
    full2 = tf.layers.dense(
        inputs=full1,
        units=10,
        activation=tf.nn.tanh,
        name="full2"
    )
    full3 = tf.layers.dense(
        inputs=full2,
        units=3,
        activation=tf.nn.sigmoid,
        name="full3"
    )
    probs = tf.nn.softmax(full3, name="pred_probs")
    pred_class = tf.argmax(input=probs, axis=1, name="pred_class")

    pred_acc = tf.reduce_mean(tf.to_float(tf.equal(pred_class, in_label)), name='pred_accuracy')

    loss = tf.losses.sparse_softmax_cross_entropy(labels=in_label, logits=probs)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss=loss)

    epoch_num = 100
    batch_size = 30

    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    sess.run(init)
    saver = tf.train.Saver()

    train_fea, train_label, eval_fea, eval_label = load_data()
    max_iter = int(epoch_num * train_label.shape[0] / batch_size)
    print("max iter: %d" % max_iter)
    idxs = list(range(0, train_label.shape[0]))
    for i in range(max_iter):
        batch_idx = random.sample(idxs, batch_size)
        batch_fea = train_fea[batch_idx]
        batch_label = train_label[batch_idx]
        _, batch_loss = sess.run((train_op, loss),
                                 feed_dict={
                                     in_fea: batch_fea,
                                     in_label: batch_label
                                 })
        if i % 10000 == 0:
            print("batch loss %f" % batch_loss)
            acc = sess.run(pred_acc,
                           feed_dict={
                               in_fea: eval_fea,
                               in_label: eval_label
                           })
            print("iter %d, with acc: %f" % (i, acc))

    acc = sess.run(pred_acc,
                   feed_dict={
                       in_fea: eval_fea,
                       in_label: eval_label
                   })
    print("train finish, with acc; %f" % (acc))

    # create data folders
    save_folder = os.path.join('../data/iris/output')
    saved_model_dir = os.path.join(save_folder, 'saved_model')
    saver_dir = os.path.join(save_folder, 'saver')
    if os.path.exists(save_folder):
        import shutil
        shutil.rmtree(save_folder, ignore_errors=True)
    if not os.path.exists(save_folder):
        print('make dir %s' % save_folder)
        os.makedirs(save_folder)
        os.makedirs(saved_model_dir)
        os.makedirs(saver_dir)

    saver_folder = os.path.join(saver_dir, 'saver_output')
    if not os.path.exists(saver_folder):
        os.makedirs(saver_folder)
    save_path = os.path.join(saver_folder, 'model')
    saver.save(sess, save_path)
    print('saver save model into %s' % save_path)

    export_dir = os.path.join(saved_model_dir, 'output')
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], clear_devices=True)
    model_path = builder.save()
    print('SavedModel save model into %s' % model_path)

    frozen_dir = os.path.join(save_folder, "frozen")
    os.mkdir(frozen_dir)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants (sess, sess.graph_def, ["pred_probs", "pred_class", "pred_accuracy"])
    frozen_graph_file = os.path.join(frozen_dir, "iris.pb")
    # Save the frozen graph
    with open (frozen_graph_file, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)