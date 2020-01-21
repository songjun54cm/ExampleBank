__author__ = "JunSong<songjun@kuaishou.com>"

# Date: 2020/1/21
"""
find all the kinds of file suffixes
"""
import os
import fire


def main(dir_path="."):
    suffixes = {}
    # r=root, d=directories, f=files
    for r, d, f in os.walk(dir_path):
        for file in f:
            sufx = file.split(".")[-1]
            suffixes[sufx] = 1
    for key in suffixes.keys():
        print(key)


if __name__ == "__main__":
    fire.Fire(main)
