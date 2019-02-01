__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/30
import argparse
import os
import tensorflow as tf


_DATA_DIR = os.path.abspath("../data/census")
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]
_HASH_BUCKET_SIZE = 1000


def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop("income_bracket")
    classes = tf.cast(tf.equal(labels, ">50K"), tf.int32)
    return features, classes

def main(config):
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    education_ind = tf.feature_column.indicator_column(education)
    edu_emb = tf.feature_column.embedding_column(education, dimension=8)

    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)
    occupation_emb = tf.feature_column.embedding_column(occupation, dimension=6)


    base_columns = [age, education_num, capital_gain, capital_loss, hours_per_week, education_ind, edu_emb, occupation_emb]
    # order: age, capital_gain, capital_loss, education, education_num, hours_per_week

    train_data_path = os.path.join(_DATA_DIR, "adult.data")
    dataset = tf.data.TextLineDataset(train_data_path)
    # dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(parse_csv, num_parallel_calls=1)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(100)

    iterator = dataset.make_one_shot_iterator()
    fea, label = iterator.get_next()
    input_fea = tf.feature_column.input_layer(fea, base_columns)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer(),
                           tf.tables_initializer())
        sess.run(init_op)
        try:
            while True:
                fea_val, label_v = sess.run([input_fea, label])
                print(fea_val.shape)
        except tf.errors.OutOfRangeError:
            print("out of range")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)