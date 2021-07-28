#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: point_net 
@Author: anzii.Luo
@Describe: 
@Date: 2021/7/28
"""
import h5py
import os
import numpy as np


def read_file_info(file_path, is_shuffle=True):
    """
    读取数据信息
    :param file_path:
    :param is_shuffle:
    :return:
    """
    with open(file=file_path) as file:
        file_info = np.array([line.rstrip() for line in file])
    idx = np.arange(0, len(file_info))
    if is_shuffle:
        np.random.shuffle(idx)
    return file_info[idx]


def load_file(file_name, is_shuffle=True):
    """
    载入数据
    :param file_name:
    :param is_shuffle: 是否打乱数据
    :return:
    """
    file = h5py.File(file_name)
    data = file["data"][:]
    label = np.squeeze(file["label"][:])

    # 是否打乱样本顺序
    if is_shuffle:
        idx = np.arange(0, len(label))
        np.random.shuffle(idx)
        # data的数据类型为np.array
        data = data[idx, ...]
        label = label[idx]

    return data, label


if __name__ == "__main__":
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data/train_files.txt"
    )
    res = read_file_info(file_path)
    print(res)
