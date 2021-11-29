#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Seven'

import cv2
from train import Model


def detect(frame):    # 用于识别测试的函数
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 图像灰化，降低计算复杂度
    cascade = cv2.CascadeClassifier(cascade_path)   # 使用人脸识别分类器，读入分类器
    faces = cascade.detectMultiScale(gray, 1.1, 5)  # 利用分类器识别出哪个区域为人脸，其实也可以不识别，但是方便起见，可以框柱图像中的人脸

    # 截取脸部图像提交给模型识别这是谁
    if len(faces) > 0:  # 如果识别到人脸
        for (x, y, w, h) in faces:  # 逐张人脸进行处理
            image = frame[y: y + h, x: x + w]   # 获得人脸具体位置
            faceID = model.face_predict(image)  # 进行识别分类
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)    # 框住人脸
            cv2.putText(frame, 's'+str(faceID+1),   # 文字提示是谁
                        (x, y+3),  # 坐标
                        cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                        1,  # 字号
                        (255, 0, 255),  # 颜色
                        2)  # 字的线宽

    return frame


if __name__ == '__main__':
    # 加载模型
    model = Model()
    model.load_model(file_path='face2.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 人脸识别分类器本地存储路径
    cascade_path = "haarcascade_frontalface_alt2.xml"

    frame1 = cv2.imread('dataset/s01/8.pgm')
    frame2 = cv2.imread('dataset/s35/1.pgm')
    frame3 = cv2.imread('dataset/s19/7.pgm')
    frame4 = cv2.imread('dataset/s04/8.pgm')

    frame1 = detect(frame1)
    frame2 = detect(frame2)
    frame3 = detect(frame3)
    frame4 = detect(frame4)

    cv2.resizeWindow("frame1", 300, 600)
    cv2.imshow("frame1", frame1)
    cv2.resizeWindow("frame2", 300, 600)
    cv2.imshow("frame2", frame2)
    cv2.resizeWindow("frame3", 300, 600)
    cv2.imshow("frame3", frame3)
    cv2.resizeWindow("frame4", 300, 600)
    cv2.imshow("frame4", frame4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


