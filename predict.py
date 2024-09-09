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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
if __name__ == "__main__":
    data_path = 'E:\DATASET\McMaster' # file.png
    model_weight='best_2.pth'
    input_shape = [256, 256, 3]
    CFA_pattern = "bayer_rggb"
    hidden_dim = 24
    layers = 4
    bilinear = False

    cfa = CFA(input_shape)
    mask = cfa.choose(CFA_pattern)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model(n_channels=3, n_classes=3, mask=mask, dim=hidden_dim, layers=layers, bilinear=bilinear)
    net.to(device=device)
    net.load_state_dict(torch.load(model_weight, map_location=device))
    net.mask = net.mask.to(device=device, dtype=torch.float32)
    net.eval()
    tests_path = glob.glob(data_path+"/*.png")
    psnr = 0
    mse = 0
    ssim = 0
    for test_path in tests_path:
        img = cv2.imread(test_path)
        img = random_crop(img, input_shape[0], eval=True)
        img = img[:, :, ::-1]  # RGB
        label = img
        img = img * mask
        img = np.transpose(img, (2, 0, 1))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        hiddn = net(img_tensor, mask)
        pred = hiddn[-1]
        # 提取结果
        pred = np.array(pred.data.cpu()[0]) * 255
        pred = np.transpose(pred, (1, 2, 0)).clip(0, 255).astype(np.uint8)

        # if you want to show the result as a image, use this
        #plt.imshow(image_np)
        #plt.show()
        psnr += PSNR(pred, label)
        mse += MSE(pred, label)
        ssim += SSIM(pred, label, channel_axis=2, data_range=255)
    print(psnr / len(tests_path))
    print(mse / len(tests_path))
    print(ssim / len(tests_path))
