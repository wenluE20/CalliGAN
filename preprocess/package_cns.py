# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import glob
import os
import pickle
import random


def pickle_examples(paths, train_path, val_path, train_val_split=0.1):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for p in paths:
                base = os.path.splitext(os.path.basename(p))[0]  # 去掉 .jpg
                parts = base.split("_")
                if len(parts) >= 2:
                    cns_code = parts[0]
                    label = int(parts[1])
                else:
                    # 兼容 "0001.jpg"：用编号当 cns_code，label 默认 0
                    cns_code = parts[0]
                    label = 0

                with open(p, 'rb') as f:
                    if cns_code == 'None':
                        print("None alert! ")
                    print("img %s" % p, label)
                    print("cns code: ", cns_code)
                    img_bytes = f.read()
                    r = random.random()
                    example = (cns_code, label, img_bytes)
                    if r < train_val_split:
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)


parser = argparse.ArgumentParser(description='Compile list of images into a pickled object for training')
parser.add_argument('--dir', dest='dir', required=True, help='path of examples')
parser.add_argument('--save_dir', dest='save_dir', required=True, help='path to save pickled files')
parser.add_argument('--split_ratio', type=float, default=0.1, dest='split_ratio',
                    help='split ratio between train and val')
args = parser.parse_args()

if __name__ == "__main__":
    train_path = os.path.join(args.save_dir, "cns_train.obj")
    val_path = os.path.join(args.save_dir, "cns_test.obj")
    # 查找所有子目录中的.jpg文件
    all_image_paths = []
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith(".jpg"):
                all_image_paths.append(os.path.join(root, file))
    pickle_examples(sorted(all_image_paths), train_path=train_path, val_path=val_path,
                    train_val_split=args.split_ratio)

