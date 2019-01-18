__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/18
import argparse


def main(config):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)