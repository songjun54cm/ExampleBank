"""
Author: songjun
Date: 2018/5/4
Description:
Usage:
"""
import argparse
import librosa
import numpy as np
def main(config):
    y, sr = librosa.load(config['file'], sr=None, mono=False, dtype=np.int16)
    y_shape = y.shape
    i = 0
    j = 0
    while i < y_shape[1] and j < 10:
        flag = False
        for c in range(y_shape[0]):
            if y[c][i] != 0:
                flag = True
            if flag:
                print(y[c][i]),
                print(" "),
        if flag:
            print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)