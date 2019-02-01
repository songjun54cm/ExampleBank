__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/2/1
import tensorflow as tf

color_data = {'color': [['G'], ['B'], ['B'], ['R']]}
color_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
)
color_embedding = tf.feature_column.embedding_column(color_column)
color_embedding_dense = tf.feature_column.input_layer(color_data, [color_embedding])
