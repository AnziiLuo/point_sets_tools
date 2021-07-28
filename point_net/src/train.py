#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: point_net 
@Author: anzii.Luo
@Describe: 
@Date: 2021/7/27
"""
import os
import numpy as np
import tensorflow as tf

from models import PointNetModel
from utils import read_file_info, load_file


@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        model = PointNetModel(data)
        loss = model.get_loss(labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer = tf.keras.optimizers.Adam()
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def one_epoch(num_point=1024, batch_size=4):
    """
    单次训练
    :param num_point: 每个样本点数（降采样）
    :param batch_size: 每批次数量
    :return:
    """
    # 载入数据
    data_base_dir = os.path.join(os.path.dirname(__file__), "data")
    train_files_path = os.path.join(data_base_dir, "train_files.txt")
    train_files_info = read_file_info(file_path=train_files_path, is_shuffle=True)

    # 训练过程
    for train_file_name in train_files_info:
        train_file_name = os.path.join(data_base_dir, train_file_name)
        train_data, train_label = load_file(file_name=train_file_name, is_shuffle=True)
        train_data = train_data[:, 0:num_point, :]
        file_size = train_label.shape[0]
        num_batches = file_size // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            point_cloud = train_data[start_idx:end_idx, :, :]
            point_cloud_label = [np.int(x) for x in train_label[start_idx:end_idx]]
            train_step(data=point_cloud, labels=point_cloud_label)


if __name__ == "__main__":
    one_epoch()
