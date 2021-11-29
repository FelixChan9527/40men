from random import random
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize([0], [1])])

def get_data(path):     # 获取数据
    i = 0
    train_imgs, train_labels = [], []
    valid_imgs, valid_labels = [], []
    test_imgs, test_labels = [], []
    for man in os.listdir(path):
        full_dir = os.path.join(path, man)
        pic = 0
        j = 0
        for pic in os.listdir(full_dir):
            pic_dir = os.path.join(full_dir, pic)
            image = Image.open(pic_dir)
            if j <= 1:  # 前两张为验证集
                image = transform(image)
                valid_imgs.append(image)
                valid_labels.append(i)
            elif j <= 3:    #三四张为测试集
                image = transform(image)
                test_imgs.append(image)
                test_labels.append(i)
            else:
                image = transform(image)
                train_imgs.append(image)
                train_labels.append(i)
            j += 1
        i += 1
    images = [train_imgs, valid_imgs, test_imgs]
    labels = [train_labels, valid_labels, test_labels]
    return images, labels

class GetDataset(Dataset):      # 生成数据集
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels
    
    def __getitem__(self, index: int):
        data = self.datas[index]
        label = self.labels[index]
        return data, label
    
    def __len__(self):
        return len(self.datas)

