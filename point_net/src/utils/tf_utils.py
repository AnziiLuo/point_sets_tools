#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: point_net 
@Author: anzii.Luo
@Describe: 
@Date: 2021/7/2
"""
import tensorflow as tf


def net_conv2d(input_data: tf.Tensor, num_output_channels, kernel_size, strides):
    """
    二维卷积操作
    :param input_data:
    :param num_output_channels: 输出的通道数
    :param kernel_size: 卷积核大小
    :param strides: 步长，[1, stride_h, stride_w, 1]
    :return:
    """
    kernel_h, kernel_w = kernel_size

    # 输入数据通道
    num_input_channels = input_data.get_shape()[-1]

    # 卷积核shape
    kernel_shape = [kernel_h, kernel_w, num_input_channels, num_output_channels]
    # 初始化核
    kernel = tf.keras.layers.Conv2D()
