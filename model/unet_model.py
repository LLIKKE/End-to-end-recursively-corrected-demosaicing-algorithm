""" Full assembly of the parts to form the complete network """
import numpy as np
from matplotlib import pyplot as plt

"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

from .unet_parts import *
from scipy.signal import convolve2d


class BilinearInterpolation(nn.Module):
    def __init__(self):
        super(BilinearInterpolation, self).__init__()
        # Define convolution kernels as parameters
        self.k_g = nn.Parameter(torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32) / 4,
                                requires_grad=False)
        self.k_r_b = nn.Parameter(torch.tensor([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=torch.float32) / 4,
                                  requires_grad=False)

    def forward(self, mosic, mask):
        batchsize, c, h, w = mosic.shape

        # Convert to PyTorch tensors and ensure the kernel dimensions match
        k_g = self.k_g.unsqueeze(0).unsqueeze(0).expand(1, 1, 3, 3)  # [1, 1, 3, 3]
        k_r_b = self.k_r_b.unsqueeze(0).unsqueeze(0).expand(1, 1, 3, 3)  # [1, 1, 3, 3]

        # Initialize output tensor
        conv_in = torch.zeros_like(mosic, dtype=torch.float32)

        # Green channel interpolation
        g = mosic[:, 1, :, :]  # Extract the green channel
        convg = F.conv2d(g.unsqueeze(1).float(), k_g, padding=1).squeeze(1)  # [batchsize, 1, h, w]
        g = g + convg

        # Red channel interpolation
        r = mosic[:, 0, :, :]  # Extract the red channel
        convr1 = F.conv2d(r.unsqueeze(1).float(), k_r_b, padding=1).squeeze(1)  # [batchsize, 1, h, w]
        convr2 = F.conv2d((r + convr1).unsqueeze(1).float(), k_g, padding=1).squeeze(1)  # [batchsize, 1, h, w]
        r = r + convr1 + convr2

        # Blue channel interpolation
        b = mosic[:, 2, :, :]  # Extract the blue channel
        convb1 = F.conv2d(b.unsqueeze(1).float(), k_r_b, padding=1).squeeze(1)  # [batchsize, 1, h, w]
        convb2 = F.conv2d((b + convb1).unsqueeze(1).float(), k_g, padding=1).squeeze(1)  # [batchsize, 1, h, w]
        b = b + convb1 + convb2

        # Stack the channels back together
        conv_in[:, 0, :, :] = r
        conv_in[:, 1, :, :] = g
        conv_in[:, 2, :, :] = b

        # Adjust the mask to apply the interpolated values only where needed
        mask = 1 - mask
        #print(conv_in.shape,mask.shape)

        # Combine the original mosaic with the interpolated values
        conv_in = mosic + conv_in * mask

        return torch.clamp(conv_in, 0, 255) / 255  #.to(torch.uint8)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dim=32, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        channel = dim
        self.inc = DoubleConv(n_channels, channel)
        self.down1 = Down(channel, channel * 2)
        self.down2 = Down(channel * 2, channel * 4)
        self.down3 = Down(channel * 4, channel * 8)
        self.down4 = Down(channel * 8, channel * 8)
        self.up1 = Up(channel * 16, channel * 4, bilinear)
        self.up2 = Up(channel * 8, channel * 2, bilinear)
        self.up3 = Up(channel * 4, channel, bilinear)
        self.up4 = Up(channel * 2, channel, bilinear)
        self.outc = OutConv(channel, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Ground_truth_fill(nn.Module):
    def __init__(self):
        super(Ground_truth_fill, self).__init__()
        self.activate = nn.Tanh()

    def forward(self, x, inter, mask):
        demask = 1 - mask
        out = self.activate(x) * demask + inter  # * self.mask # [0,1]+[0,1]*mask
        return out, self.activate(x) * demask


class Reconstruction(nn.Module):
    def __init__(self, n_channels, n_classes, dim, bilinear):
        super(Reconstruction, self).__init__()
        self.unet = UNet(n_channels, n_classes, dim, bilinear)
        self.GTF = Ground_truth_fill()

    def forward(self, x, inter, mask):
        out = self.unet(x)
        #out = x + out
        return self.GTF(out, inter, mask)


class model(nn.Module):
    def __init__(self, n_channels, n_classes, mask, dim=64, layers=1, bilinear=False):
        super(model, self).__init__()

        mask = torch.from_numpy(mask)
        self.mask = mask.permute(2, 0, 1)
        self.inter = BilinearInterpolation()
        self.layers = nn.ModuleList([Reconstruction(n_channels, n_classes, dim, bilinear) for i in range(layers)])

    def bilinear(self, mosic, mask):
        r = mosic[:, :, 0]
        g = mosic[:, :, 1]
        b = mosic[:, :, 2]
        # green interpolation
        k_g = 1 / 4 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        convg = convolve2d(g, k_g, 'same')
        g = g + convg

        # red interpolation
        k_r_1 = 1 / 4 * np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        convr1 = convolve2d(r, k_r_1, 'same')
        convr2 = convolve2d(r + convr1, k_g, 'same')
        r = r + convr1 + convr2

        # blue interpolation
        k_b_1 = 1 / 4 * np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        convb1 = convolve2d(b, k_b_1, 'same')
        convb2 = convolve2d(b + convb1, k_g, 'same')
        b = b + convb1 + convb2

        conv_in = np.vstack((r, g, b))  # [c,h,w]

        mask = 1 - mask  # 将真值去掉
        conv_in = mosic + conv_in * mask  # 将插值的真值去掉，变为掠阵后的
        return conv_in.clip(0, 255).astype(np.uint8)

    def forward(self, x, mask):
        inter = self.inter(x, self.mask)  #.detach()
        hidden = []
        out = inter
        for layer in self.layers:
            out, p = layer(out, inter, self.mask)
            hidden.append(p)
        return hidden


if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)
