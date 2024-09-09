import numpy as np
import torch
import cv2
import os
import glob

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import random
from augs import random_crop


class DATA_Loader(Dataset):
    def __init__(self, data_path,mask, size=None):
        self.size = size
        if size is None:
            self.size = [500, 500, 3]
        self.data_path = data_path
        self.imgs_path = glob.glob(data_path+r'/*.png')
        self.mask = mask
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = random_crop(image,self.size[0])
        if flipCode != 2:
            flip = cv2.flip(flip, flipCode)
        return flip

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        image = cv2.imread(image_path)

        flipCode = random.choice([-1, 0, 1, 2]) # 数据增强，区域裁剪和翻转,[h,w,c]
        image = self.augment(image, flipCode)

        image = image[:,:,::-1].copy()  # reshape RGB [h,w,c]

        #plt.imshow(image/255)
        #plt.show()
        label = np.transpose(image, (2, 0, 1)) # label由原始图像变为[c,h,w]

        image = image * self.mask # CFA列阵过滤
        image = np.transpose(image, (2, 0, 1)) # ->[c,h,w]
        if label.max() > 1:
            label = label / 255
            #image = image / 255

        return image, label

    def __len__(self):
        return len(self.imgs_path)

