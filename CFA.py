import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class CFA:
    def __init__(self,im_shape,pattern="bayer_rggb"):
        self.im_shape=im_shape
        self.pattern=pattern
    def create_bayer_mask(self,shape):
        mask = np.zeros(shape)
        # Red channel
        mask[0::2, 0::2, 0] = 1

        # Green channel
        mask[0::2, 1::2, 1] = 1
        mask[1::2, 0::2, 1] = 1

        # Blue channel
        mask[1::2, 1::2, 2] = 1

        return mask  # np.dstack((r_mask, g_mask, b_mask))

    def create_xtrans_mask(self,shape):
        base_g_mask = np.array([
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 1]
        ])

        base_r_mask = np.array([
            [0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 0]
        ])

        base_b_mask = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])

        g_mask = np.tile(base_g_mask, (shape[0] // 6 + 1, shape[1] // 6 + 1))[:shape[0], :shape[1]]
        r_mask = np.tile(base_r_mask, (shape[0] // 6 + 1, shape[1] // 6 + 1))[:shape[0], :shape[1]]
        b_mask = np.tile(base_b_mask, (shape[0] // 6 + 1, shape[1] // 6 + 1))[:shape[0], :shape[1]]

        return np.dstack((r_mask, g_mask, b_mask))

    def quad_bayer_mask(self,shape):
        # 创建四像素块的Bayer CFA矩阵
        quad_bayer_cfa = np.array([
            ["R", "R", "G", "G"],
            ["R", "R", "G", "G"],
            ["G", "G", "B", "B"],
            ["G", "G", "B", "B"]
        ])

        cfa_shape = quad_bayer_cfa.shape
        # 初始化R, G, B三个通道
        mask = np.zeros(shape)

        # 逐像素比对并提取对应通道的值
        for i in range(shape[0]):
            for j in range(shape[1]):
                if quad_bayer_cfa[i % cfa_shape[0], j % cfa_shape[1]] == "R":
                    mask[i, j, 0] = 1
                elif quad_bayer_cfa[i % cfa_shape[0], j % cfa_shape[1]] == "G":
                    mask[i, j, 1] = 1
                elif quad_bayer_cfa[i % cfa_shape[0], j % cfa_shape[1]] == "B":
                    mask[i, j, 2] = 1

        return mask
    def choose(self,pattern=None):
        if pattern is not None:
            self.pattern = pattern
        if self.pattern == 'bayer_rggb':
            return self.create_bayer_mask(self.im_shape)
        elif self.pattern == 'xtrans':
            return self.create_xtrans_mask(self.im_shape)
        elif self.pattern == 'quad_bayer':
            return self.quad_bayer_mask(self.im_shape)
        else:
            raise NotImplementedError('Only bayer_rggb, quad_bayer and xtrans are implemented')


if __name__ == '__main__':

    raw = cv.imread(r'E:\DATASET\McMaster\1.png')
    print(f"raw shape: {raw.shape}, max: {np.max(raw)}, min: {np.min(raw)}")
    raw = raw[:,:,::-1]
    plt.imshow(raw)
    plt.title("raw")
    plt.show()
    #plt.savefig("raw.png")

    cfa = CFA(raw.shape)
    mask = cfa.choose("bayer_rggb")
    mask = mask.astype(np.uint8)
    print("mask\n",mask[:5,:5,0],"\n-----\n",mask[:5,:5,1],"\n-----\n",mask[:5,:5,2])
    print("mask shape ",mask.shape)
    image_mosaic = mask * raw

    plt.imshow(image_mosaic)
    plt.title("mosaic")
    plt.show()
