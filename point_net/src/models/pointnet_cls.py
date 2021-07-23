#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: point_net 
@Author: anzii.Luo
@Describe: 
@Date: 2021/6/30
"""
import h5py
import tensorflow as tf

from src.common import input_transform_net, feature_transform_net


class PointNetModel:
    def __init__(self, point_cloud: tf.Tensor):
        self.point_cloud = point_cloud

    def get_model(self):

        # 输入点云转换
        transform = input_transform_net(point_cloud=self.point_cloud, out_dim=3)
        point_cloud_transform = tf.matmul(self.point_cloud, transform)

        # 增加一个维度用于卷积计算
        input_image = tf.expand_dims(input=point_cloud_transform, axis=-1)
        # 两个卷积层
        net = tf.keras.layers.Conv2D(
            filters=64, kernel_size=[1, 3], strides=[1, 1], padding="valid"
        )(input_image)
        net = tf.keras.layers.Conv2D(
            filters=64, kernel_size=[1, 1], strides=[1, 1], padding="valid"
        )(net)

        feature_transform = feature_transform_net(input_data=net, out_dim=64)


if __name__ == "__main__":
    h5_filename = "../data/ply_data_train0.h5"
    f = h5py.File(h5_filename)
    data = f["data"][:1]
    point_cloud_data = tf.constant(data)
    PointNetModel(point_cloud=point_cloud_data).get_model()
    # print(transform_res)
