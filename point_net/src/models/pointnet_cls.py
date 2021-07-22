#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: point_net 
@Author: anzii.Luo
@Describe: 
@Date: 2021/6/30
"""
import tensorflow as tf

from src.common import input_transform_net


class PointNetModel:
    def __init__(self, point_cloud: tf.Tensor):
        self.point_cloud = point_cloud

    def get_model(self):

        # 输入点云转换
        transform = input_transform_net(point_cloud=self.point_cloud, out_dim=3)
        point_cloud_transform = tf.matmul(self.point_cloud, transform)
