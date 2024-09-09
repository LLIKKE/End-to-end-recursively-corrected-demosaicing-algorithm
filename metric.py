import numpy as np
import torch
from skimage.metrics import structural_similarity as SSIM
from torch import nn


def PSNR(image1, image2):
    img1 = image1
    img2 = image2
    if img1.shape != img2.shape:
        raise ValueError("输入的图像大小不一致")
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def MSE(img1, img2):
    loss_map = (img1 - img2) ** 2
    return np.mean(loss_map)

class MaskedMSELoss(nn.Module):
    def __init__(self,mask):
        super(MaskedMSELoss, self).__init__()
        self.mask = mask
        self.non_zero_count = torch.count_nonzero(self.mask)
    def forward(self, img1, img2):
        #print(img1.shape,mask.shape)
        loss_map = (img1 - img2) ** 2
        # 计算有掩码的均方误差
        masked_loss = torch.sum(loss_map * self.mask)
        return masked_loss
