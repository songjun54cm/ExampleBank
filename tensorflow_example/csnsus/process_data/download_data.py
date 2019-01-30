__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/25
import argparse
import os
from six.moves import urllib
import tensorflow as tf

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)


def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format."""
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.gfile.Open(temp_file, 'r') as temp_eval_file:
        with tf.gfile.Open(filename, 'w') as eval_file:
            for line in temp_eval_file:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += '\n'
                eval_file.write(line)
    tf.gfile.Remove(temp_file)


def main(config):
    data_dir = "data/census"
    tf.gfile.MakeDirs(data_dir)
    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    _download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    _download_and_clean_file(eval_file_path, EVAL_URL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)

