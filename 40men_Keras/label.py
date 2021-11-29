#!/usr/bin/env python
# -*- coding: utf-8 -*-
#     对图像进行初步加工，将
__author__ = 'Seven'
import os
import numpy as np
import cv2

# 定义图片尺寸
IMAGE_SIZE = 90


# 按照定义图像大小进行尺度调整
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = 0, 0, 0, 0
    # 获取图像尺寸
    h, w, _ = image.shape
    # 找到图片最长的一边
    longest_edge = max(h, w)
    # 计算短边需要填充多少使其与长边等长
    if h < longest_edge:
        d = longest_edge - h
        top = d // 2    # 整除
        bottom = d // 2 # 整除
    elif w < longest_edge:
        d = longest_edge - w
        left = d // 2   # 整除
        right = d // 2  # 整除
    else:
        pass

    # 设置填充颜色，将图片数据变成长宽大小一样的矩阵
    BLACK = [0, 0, 0]
    # 对原始图片进行填充操作
    constant = cv2.copyMakeBorder(image, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))

images, labels = list(), list()
# 读取训练数据
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        # 合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        # 如果是文件夹，则继续递归调用
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.pgm'):   # 如果文件以jpg结尾，则文件就是图片
                image = cv2.imread(full_path)   # 读取文件夹图片
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)# 将图像整型至64*64
                images.append(image)    # 将每张图片的像素矩阵塞进images中，以便用来进行训练
                labels.append(path_name)    # 同时将文件夹名字作为标签，塞进label矩阵中
    return images, labels   # 返回这两个矩阵


# 从指定路径读取训练数据
def load_dataset(path_name):
    images, labels = read_path(path_name)
    # 由于图片是基于矩阵计算的， 将其转为矩阵
    print(labels)
    images = np.array(images)   # 将图像数据转为可以在numpy进行运算的矩阵
    print(images.shape)

    label_length = len(labels)
    lab = np.zeros(label_length)
    i = 0
    j = 0
    for label in labels:
        if i != 0:
            if i % 10 == 0:
                j = j + 1
        lab[i] = j
        print('lab=', lab[i], '  label = ', label)
        i = i+1
    labels = lab

    return images, labels


####################################################################################
####################################################################################


if __name__ == '__main__':
    images, labels = load_dataset('dataset')    # 读取照片进行预处理，统一成64大小的图像矩阵数据
    print(labels)
    print('load over')
