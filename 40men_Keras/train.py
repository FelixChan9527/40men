#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Seven'
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
from label import load_dataset, resize_image, IMAGE_SIZE
import warnings
warnings.filterwarnings('ignore')


Classes = 40    # 种类数据
epoch = 1000    # 训练迭代次数
batch = 32      # 批次大小

class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None
        # 测试集
        self.test_images = None
        self.test_labels = None
        # 数据加载路径
        self.path_name = path_name
        # 网络输入维度
        self.input_shape = None

    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=Classes):
        # 加载数据集至内存，输入原始图像行列大小，3维图像通道，分类为2种
        images, labels = load_dataset(self.path_name)   # 加载图像
        # 将训练用的图像又分为一部分是验证集，一部分是训练集，验证集不参与训练，验证集大小为0.3倍
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels,
                                                                                test_size=0.3,
                                                                                random_state=random.randint(0, 10))
        # 改变训练集验证集维度
        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
        self.input_shape = (img_rows, img_cols, img_channels)
        # 输出训练集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')
        # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
        # one-hot编码就是将标签变成二进制
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)
        # 像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        # 将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255.0
        test_images /= 255.0
        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels


# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None

    # 建立模型函数
    def build_model(self, dataset, nb_classes=Classes):
        self.model = Sequential()   # 逐层加网络

        # 添加CNN网络层，一共18层
        self.model.add(Conv2D(32, 3, 3, border_mode='valid', input_shape=dataset.input_shape))  # 1 二维卷积层
        self.model.add(Activation('relu'))  # 2 激活函数层，激活函数增加非线性度
        self.model.add(Conv2D(32, 3, 3))  # 3 2维卷积层：第一个数为滤波器数量即输出的维度，第二个数为卷积核的大小，第三个为步长
        self.model.add(Activation('relu'))  # 4 激活函数层，激活函数为relu
        self.model.add(MaxPool2D(pool_size=(2, 2)))  # 5 池化层
        self.model.add(Dropout(0.25))  # 6 Dropout层，将部分神经元屏蔽，节省资源并且提高不确定性
        self.model.add(Flatten())  # 7 Flatten层，用于和全连接层的过渡
        self.model.add(Dense(400))  # 8 Dense层,又被称作全连接层，可以看做BP算法
        self.model.add(Activation('relu'))  # 9 激活函数层
        self.model.add(Dropout(0.5))  # 10 Dropout层
        self.model.add(Dense(nb_classes))  # 11 Dense层
        self.model.add(Activation('softmax'))  # 12 分类层，输出最终结果，此处的激活函数用softmax，用于最后的神经元归类

        # 输出模型概况
        self.model.summary()

    # 训练模型
    # batch_size为一次训练的样本数；迭代50次

    def train(self, dataset, batch_size=batch, nb_epoch=epoch):
        # 采用SGD+momentum的优化器进行训练，lr为学习率，decay为衰减率，
        # momentum，梯度动量下降法的动量参数
        # 使用nesterov动量
        sgd = Adam(lr=0.001, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作

        # 定义数据生成器用于数据扩增
        datagen = ImageDataGenerator(
            featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
            samplewise_center=False,  # 是否使输入数据的每个样本均值为0
            featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
            samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
            zca_whitening=False,  # 是否对输入数据施以ZCA白化
            rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
            width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
            height_shift_range=0.2,  # 同上，只不过这里是垂直
            horizontal_flip=True,  # 是否进行随机水平翻转
            vertical_flip=False)  # 是否进行随机垂直翻转

        # 利用生成器开始训练模型，输入训练数据验证数据以及标签数据，训练批次，迭代次数
        self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                              batch_size=batch_size),
                                 samples_per_epoch=dataset.train_images.shape[0],
                                 nb_epoch=nb_epoch,
                                 validation_data=(dataset.test_images, dataset.test_labels))

    MODEL_PATH = 'face2.h5'

    # 保存模型函数
    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    # 加载模型函数
    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    # 识别人脸
    def face_predict(self, image):
        image = resize_image(image)     # 整形图片
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        # 浮点并归一化，将图片像素变为0~1的范围内
        image = image.astype('float32')
        image /= 255.0

        # 给出输入属于各个类别的概率，则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict_proba(image)

        # 给出类别预测：0~39
        result_class = self.model.predict_classes(image)
        print("The class is: s", str(result_class[0]+1),    # 输出种类，以及准确率
              ' | Accuracy:', int((result[0, result_class[0]])*100), "%")

        # 返回类别预测结果
        return result_class[0]


if __name__ == '__main__':
    dataset = Dataset('face')   # 加载数据集
    dataset.load()

    model = Model()             # 创建模型
    model.build_model(dataset)
    model.train(dataset)    # 训练模型
    model.save_model(file_path='face2.h5')  # 保存模型





