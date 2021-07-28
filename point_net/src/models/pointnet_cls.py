#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: point_net 
@Author: anzii.Luo
@Describe: 
@Date: 2021/6/30
"""
from abc import ABC

import h5py
import tensorflow as tf

from src.common import input_transform_net, feature_transform_net


class PointNetModel(tf.keras.Model, ABC):
    def __init__(self, **kwargs):
        super(PointNetModel, self).__init__(**kwargs)

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

        # 特征转换，out_dim == filters
        feature_transform = feature_transform_net(input_data=net, out_dim=64)
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), feature_transform)
        net_transformed = tf.expand_dims(net_transformed, axis=[2])

        # 卷积计算
        net = tf.keras.layers.Conv2D(
            filters=64, kernel_size=[1, 1], strides=[1, 1], padding="valid"
        )(net_transformed)
        net = tf.keras.layers.Conv2D(
            filters=128, kernel_size=[1, 1], strides=[1, 1], padding="valid"
        )(net)
        net = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=[1, 1], strides=[1, 1], padding="valid"
        )(net)
        net = tf.keras.layers.MaxPooling2D(
            pool_size=[num_point, 1], strides=[1, 1], padding="valid"
        )(net)
        net = tf.reshape(net, [batch_size, -1])
        net = tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(net)
        net = tf.keras.layers.Dropout(rate=0.3)(net, training=True)
        net = tf.keras.layers.Dense(
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(net)
        net = tf.keras.layers.Dropout(rate=0.3)(net, training=True)
        net = tf.keras.layers.Dense(
            units=40,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )(net)
        self.pre = net

    def call(self, point_cloud, training=None, mask=None):
        batch_size = self.point_cloud.get_shape()[0]
        num_point = self.point_cloud.get_shape()[1]

    @tf.function
    def get_loss(self, labels):
        """
        计算准确率
        :param labels:
        :return:
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.pre, labels=labels
        )
        classify_loss = tf.reduce_mean(loss)

        return classify_loss


if __name__ == "__main__":
    import numpy as np

    index = 2
    h5_filename = "../data/ply_data_train0.h5"
    f = h5py.File(h5_filename)
    data = f["data"][:index]
    label = [np.int(x) for x in (tf.reshape(f["label"][:index], -1))]
    label = tf.constant(label)
    point_cloud_data = tf.constant(data)
    point_net_model = PointNetModel(point_cloud=point_cloud_data)
    loss = point_net_model.get_loss(labels=label)
    # # print(loss)
