#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: point_net 
@Author: anzii.Luo
@Describe: 输入数据转换网络
@Date: 2021/6/30
"""
import h5py

import numpy as np
import tensorflow as tf


def input_transform_net(point_cloud: tf.Tensor, out_dim=3):
    """
    输入转换网络
    :param point_cloud: 点云数据，维度 BxNx3
    :param out_dim: 输出维度 3*out_dim
    :return:
    """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]

    # 为进行卷积计算，扩充一个维度
    input_image = tf.expand_dims(input=point_cloud, axis=-1)
    net = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[1, 3],
        strides=[1, 1],
        padding="valid",
    )(input_image)
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
    net = tf.keras.layers.Dense(
        units=256,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )(net)
    initializer = tf.keras.initializers.Constant(0.0)
    weights = tf.Variable(
        initial_value=initializer(shape=(256, 3 * out_dim)),
        name="weights_input",
        dtype=tf.float32,
    )
    biases = tf.Variable(
        initial_value=initializer(shape=(3 * out_dim, 1)),
        name="biases_input",
        dtype=tf.float32,
    )
    biases = tf.Variable(tf.squeeze(biases, axis=[1]))
    biases = biases.assign_add(
        tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
    )
    transform = tf.matmul(net, weights)
    transform = tf.nn.bias_add(transform, biases)
    transform = tf.reshape(transform, [batch_size, 3, out_dim])

    return transform


def feature_transform_net(input_data: tf.Tensor, out_dim=64):
    """
    特征转换网络
    :param input_data: 输入数据，维度 BxNx1xK
    :param out_dim: 输出维度 KxK
    :return:
    """
    batch_size = input_data.get_shape()[0]
    num_point = input_data.get_shape()[1]

    net = tf.keras.layers.Conv2D(
        filters=64, kernel_size=[1, 1], strides=[1, 1], padding="valid"
    )(input_data)
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
    net = tf.keras.layers.Dense(
        units=256,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform(),
    )(net)
    initializer = tf.keras.initializers.constant(0.0)
    weights = tf.Variable(
        initial_value=initializer(shape=(256, out_dim * out_dim)),
        name="weights_feature",
        dtype=tf.float32,
    )
    biases = tf.Variable(
        initial_value=initializer(shape=(out_dim * out_dim, 1)),
        name="biases_feature",
        dtype=tf.float32,
    )
    biases = tf.Variable(tf.squeeze(biases, axis=[1]))
    biases = biases.assign_add(
        tf.constant(np.reshape(np.eye(out_dim), -1), dtype=tf.float32)
    )
    transform = tf.matmul(net, weights)
    transform = tf.nn.bias_add(transform, biases)
    transform = tf.reshape(transform, [batch_size, out_dim, out_dim])

    return transform


if __name__ == "__main__":
    h5_filename = "../data/ply_data_train0.h5"
    f = h5py.File(h5_filename)
    data = f["data"][:1]
    point_cloud_data = tf.constant(data)
    transform_res = input_transform_net(point_cloud=point_cloud_data)
    # print(transform_res)
