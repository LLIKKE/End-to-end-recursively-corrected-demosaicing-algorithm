import glob
import time

import numpy as np
import torch
import os
import cv2
from matplotlib import pyplot as plt

from augs import random_crop
from metric import *
from CFA import CFA
from model.unet_model import UNet, model
import os
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == "__main__":

    data_path = 'E:\DATASET\Kodak24'
    input_shape = [256, 256, 3]
    CFA_pattern = "bayer_rggb"

    hidden_dim = 24
    layers = 4
    bilinear = False

    cfa = CFA(input_shape)
    mask = cfa.choose(CFA_pattern)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = model(n_channels=3, n_classes=3, mask=mask, dim=hidden_dim, layers=layers, bilinear=bilinear)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_layer4_dim24.pth', map_location=device))
    # 测试模式
    net.mask = net.mask.to(device=device, dtype=torch.float32)
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob(data_path+"/*.png")
    # 遍历素有图片
    psnr = 0
    mse = 0
    ssim = 0
    idx=0
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        img = random_crop(img, input_shape[0], eval=True)
        # 转为灰度图
        img = img[:, :, ::-1]  # RGB
        label = img

        img = img * mask
        #plt.imshow(img.clip(0, 255).astype(np.uint8))
        #plt.show()

        img = np.transpose(img, (2, 0, 1))
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        hiddn = net(img_tensor, mask)
        lis = []
        for i in range(len(hiddn)):
            pred = hiddn[i]
            # 提取结果
            pred = np.array(pred.data.cpu()[0]) * 510
            # 处理结果
            pred = np.transpose(pred, (1, 2, 0)).clip(0, 255).astype(np.uint8)
            #plt.imshow(pred)
            #plt.show()
            lis.append(pred)
        result = np.hstack(lis)
        #plt.imshow(result)
        print(f'mcmaster_result\image_{idx}.png')
        result_img = Image.fromarray(result)
        result_img.save(f'kodak24_fit\image_{idx}.png')
        idx+=1
        #plt.show()

