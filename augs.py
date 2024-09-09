import cv2
import numpy as np


def random_crop(image, crop_size,eval=False):
    """
    随机裁剪图像为指定尺寸。

    参数:
        image (numpy.ndarray): 输入的图像（彩色或灰度图像）。
        crop_size (int or tuple): 目标裁剪尺寸。如果是整数，则表示裁剪后的宽度和高度相同。
                                  如果是元组，则表示 (crop_height, crop_width)。

    返回:
        numpy.ndarray: 裁剪后的图像。
    """
    # 获取图像的高度和宽度
    height, width, _ = image.shape
    #print(height,"   ",width)
    # 如果 crop_size 是整数，设定为方形裁剪
    if isinstance(crop_size, int):
        crop_height = crop_width = crop_size
    else:
        crop_height, crop_width = crop_size
    if eval:
        np.random.seed(42)
    # 确保图像的尺寸大于裁剪尺寸
    if height >= crop_height and width >= crop_width:
        # 随机生成裁剪区域的左上角坐标
        x = np.random.randint(0, width - crop_width + 1)
        y = np.random.randint(0, height - crop_height + 1)

        # 使用切片操作裁剪图像
        cropped_image = image[y:y + crop_height, x:x + crop_width]
        return cropped_image
    else:
        raise ValueError(f"图像尺寸小于 {crop_height}x{crop_width}，无法裁剪")


