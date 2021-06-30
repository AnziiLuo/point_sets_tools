#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: point_cloud
@Author: anzii.Luo
@Describe: Spatial Transformer Networks 网络实现(对图像的仿射转换)
@Date: 2021/5/28
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cv2 import cv2


class SNT:
    def __init__(
        self,
        image_path,
        out_size,
        theta_init=np.array(
            [[0.5, 0, 0], [0, 0.5, 0]],
        ),
        num_batch=3,
    ):
        """
        初始化
        :param image_path: 输入图像地址
        :param out_size: 输出尺寸
        :param theta_init: 仿射函数初始值
        :param num_batch: 批次
        """
        image = cv2.imread(image_path) / 255
        # 把num_batch的图做成一张高维大图
        img = tf.expand_dims(image, 0)
        input_list = []
        for i in range(num_batch):
            input_list.append(img)
        self.img_batch = tf.concat(values=input_list, axis=0)

        # 仿射变换的初始值
        # 当num_batch==2：[[[0.5, 0, 0], [0, 0.5, 0]], [[0.5, 0, 0], [0, 0.5, 0]]]
        self.theta_batch = tf.tile(
            tf.expand_dims(tf.constant(theta_init), 0), [num_batch, 1, 1]
        )
        self.out_size = out_size
        self.num_batch = num_batch

    def out_grid(self):
        """
        创建输出图像网格
        :return: (num_batch, 3, out_size[0]*out_size[1])
        """
        x_out, y_out = tf.meshgrid(
            tf.linspace(-1, 1, self.out_size[1]), tf.linspace(-1, 1, self.out_size[0])
        )

        # x_out和y_out的shape是一样的
        ones = tf.ones_like(x_out)

        # 仿射计算的系数需要和图像每一个点进行计算
        # 因此需要构造[[x1, x2, ..., xn], [y1, y2, ..., yn], [1, 1, ..., 1]]
        grid = tf.concat(
            [
                tf.reshape(x_out, (1, -1)),
                tf.reshape(y_out, (1, -1)),
                tf.reshape(ones, (1, -1)),
            ],
            axis=0,
        )
        grid_batch = tf.tile(tf.expand_dims(grid, 0), [self.num_batch, 1, 1])
        return grid_batch

    def get_pixel_value(self, x, y):
        """
        根据坐标获取原图像素
        :param x: 转换图像x
        :param y: 转换图像y
        :return:
        """
        # 构建在原图上对应的坐标
        batch_index = tf.range(self.num_batch)
        batch_index = tf.reshape(batch_index, shape=(self.num_batch, 1))
        batch_index = tf.tile(batch_index, multiples=[1, tf.shape(x)[1]])
        pixel_index = tf.stack([batch_index, y, x], axis=2)

        # tf.gather_nd的取值方法是把index最后一维换成对应坐标数值，其他维度保留
        pixel = tf.gather_nd(self.img_batch, pixel_index)
        return pixel

    def gather_channel(self, trans_grid):
        """
        双线性插值计算，并对每个像素的RGB进行映射
        :param trans_grid: 仿射变换生成的网格
        :return:
        """
        # 切分x和y
        grid_x = tf.reshape(
            tf.slice(trans_grid, begin=[0, 0, 0], size=[-1, 1, -1]),
            (self.num_batch, -1),
        )
        grid_y = tf.reshape(
            tf.slice(trans_grid, begin=[0, 1, 0], size=[-1, 1, -1]),
            (self.num_batch, -1),
        )

        # 把grid的长和宽缩放到和原始图片一致
        image_x = tf.shape(self.img_batch)[2]
        image_y = tf.shape(self.img_batch)[1]
        grid_x = (grid_x + 1) * tf.cast(image_x, dtype=tf.float64) / 2
        grid_y = (grid_y + 1) * tf.cast(image_y, dtype=tf.float64) / 2

        # 将网格x和y都取整，原图上才有对应的像素
        grid_x_0 = tf.cast(grid_x, dtype=tf.int32)
        grid_y_0 = tf.cast(grid_y, dtype=tf.int32)
        grid_x_1 = grid_x_0 + 1
        grid_y_1 = grid_y_0 + 1

        # 不要超过边界
        grid_x_0 = tf.clip_by_value(
            grid_x_0, clip_value_min=0, clip_value_max=image_x - 1
        )
        grid_x_1 = tf.clip_by_value(
            grid_x_1, clip_value_min=0, clip_value_max=image_x - 1
        )
        grid_y_0 = tf.clip_by_value(
            grid_y_0, clip_value_min=0, clip_value_max=image_y - 1
        )
        grid_y_1 = tf.clip_by_value(
            grid_y_1, clip_value_min=0, clip_value_max=image_y - 1
        )

        # 获取每个网格位置对应像素值
        pixel_1 = self.get_pixel_value(grid_x_0, grid_y_0)
        pixel_2 = self.get_pixel_value(grid_x_0, grid_y_1)
        pixel_3 = self.get_pixel_value(grid_x_1, grid_y_0)
        pixel_4 = self.get_pixel_value(grid_x_1, grid_y_1)

        # expand_dims增加维度是为后面相乘计算
        weight_1 = tf.expand_dims(
            (grid_x - tf.cast(grid_x_0, dtype=tf.float64))
            * (grid_y - tf.cast(grid_y_0, dtype=tf.float64)),
            axis=2,
        )
        weight_2 = tf.expand_dims(
            (grid_x - tf.cast(grid_x_0, dtype=tf.float64))
            * (tf.cast(grid_y_1, dtype=tf.float64) - grid_y),
            axis=2,
        )
        weight_3 = tf.expand_dims(
            (tf.cast(grid_x_1, dtype=tf.float64) - grid_x)
            * (grid_y - tf.cast(grid_y_0, dtype=tf.float64)),
            axis=2,
        )
        weight_4 = tf.expand_dims(
            (tf.cast(grid_x_1, dtype=tf.float64) - grid_x)
            * (tf.cast(grid_y_1, dtype=tf.float64) - grid_y),
            axis=2,
        )

        # 双线性插值
        inter_img = tf.add_n(
            [
                weight_4 * pixel_1,
                weight_3 * pixel_2,
                weight_2 * pixel_3,
                weight_1 * pixel_4,
            ]
        )
        return inter_img

    def transform(self):
        """
        进行图像仿射变换
        :return:
        """
        # 创建输出图像网格
        grid_batch = self.out_grid()

        # 对输出图像进行仿射计算：A x (x_t, y_t, 1)^T -> (x_s, y_s)
        # trans_grid的shape：[num_batch, 2, out_size[0]*out_size[1]]
        trans_grid = tf.matmul(self.theta_batch, grid_batch)

        # 获取各个像素点在原始图像对应RGB
        inter_img = self.gather_channel(trans_grid=trans_grid)
        inter_img = tf.reshape(
            inter_img, (self.num_batch, self.out_size[0], self.out_size[1], -1)
        )
        return inter_img


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_path = os.path.join(base_dir, "test/stn_test.jpg")
    y = SNT(
        image_path=img_path,
        out_size=[600, 800],
        num_batch=3,
        theta_init=np.array(
            [[0.5, 0, 0], [1, 0.5, 1]],
        ),
    ).transform()
    plt.imshow(y[0])
    # plt.show()
    save_path = os.path.join(base_dir, "test/stn_test_res.png")
    plt.savefig(save_path)
