import random
import time

import numpy as np
import cv2
# paddle实现torchvision.transform
import paddle.vision.transforms
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter, map_coordinates


def horizontal_flip(im):
    return paddle.vision.hflip(im)


def vertical_flip(im):
    return paddle.vision.vflip(im)


def fun_color(ori_image, coefficient):
    # 色度,增强因子为1.0是原始图像
    # 色度增强 1.5
    # 色度减弱 0.8
    image = Image.fromarray(ori_image.astype('uint8')).convert('RGB')
    enh_col = ImageEnhance.Color(image)
    image_colored1 = enh_col.enhance(coefficient)
    return np.array(image_colored1)


def fun_Contrast(ori_image, coefficient):
    # 对比度，增强因子为1.0是原始图片
    # 对比度增强 1.5
    # 对比度减弱 0.8
    image = Image.fromarray(ori_image.astype('uint8')).convert('RGB')
    enh_con = ImageEnhance.Contrast(image)
    image_contrasted1 = enh_con.enhance(coefficient)
    return np.array(image_contrasted1)


def fun_Sharpness(ori_image, coefficient):
    # 锐度，增强因子为1.0是原始图片
    # 锐度增强 3
    # 锐度减弱 0.8
    image = Image.fromarray(ori_image.astype('uint8')).convert('RGB')
    enh_sha = ImageEnhance.Sharpness(image)
    image_sharped1 = enh_sha.enhance(coefficient)
    return np.array(image_sharped1)


def fun_bright(ori_image, coefficient):
    # 变亮 1.5
    # 变暗 0.8
    # 亮度增强,增强因子为0.0将产生黑色图像； 为1.0将保持原始图像。
    image = Image.fromarray(ori_image.astype('uint8')).convert('RGB')
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened1 = enh_bri.enhance(coefficient)
    return np.array(image_brightened1)

class ColorJitter(object):
    def __init__(self, bright=0.5, sharp=0.5, contrast=0.5, color=0.5):
        self.b = random.uniform(1 - bright, 1 + bright)
        self.s = random.uniform(1 - sharp, 1 + sharp)
        self.con = random.uniform(1 - contrast, 1 + contrast)
        self.col = random.uniform(1 - color, 1 + color)

    def __call__(self, im1, im2):
        im1, im2 = fun_bright(im1, self.b), fun_bright(im2, self.b)
        im1, im2 = fun_Sharpness(im1, self.s), fun_Sharpness(im2, self.s)
        im1, im2 = fun_Contrast(im1, self.con), fun_Contrast(im2, self.con)
        im1, im2 = fun_color(im1, self.col), fun_color(im2, self.col)
        return im1, im2

class CropCenter(object):
    def __init__(self, rate=0.9):
        self.rate = rate

    def __call__(self, im1, im2):
        im1 = im1[int(im1.shape[0] * (1 - self.rate)):int(im1.shape[0] * self.rate),
              int(im1.shape[1] * (1 - self.rate)):int(im1.shape[1] * self.rate)]
        im2 = im2[int(im2.shape[0] * (1 - self.rate)):int(im2.shape[0] * self.rate),
              int(im2.shape[1] * (1 - self.rate)):int(im2.shape[1] * self.rate)]
        return im1, im2


# 随机裁剪 出img_size的图片
class RandomCrop(object):
    def __init__(self, img_size):
        if isinstance(img_size, int):
            self.img_width, self.img_height = img_size, img_size
        else:
            self.img_height, self.img_width = img_size[0], img_size[1]

    def __call__(self, im1, im2):
        return self.Random_crop(im1.copy(), im2.copy())

    def Random_crop(self, im1, im2):
        height, width, _ = im1.shape
        width_range = width - self.img_width
        height_range = height - self.img_height
        try:
            random_ws = np.random.randint(width_range)
            random_hs = np.random.randint(height_range)
            random_wd = self.img_width + random_ws
            random_hd = self.img_height + random_hs
            im1 = im1[random_hs:random_hd, random_ws:random_wd]
            im2 = im2[random_hs:random_hd, random_ws:random_wd]
            return im1, im2
        except:
            return Resize((self.img_height, self.img_width))(im1, im2)


class Compose:

    def __init__(self, transforms, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, im1, im2):

        if isinstance(im1, str):
            im1 = cv2.imread(im1).astype('float32')
        if isinstance(im2, str):
            im2 = cv2.imread(im2).astype('float32')
        if im1 is None or im2 is None:
            raise ValueError('Can\'t read The image file {} and {}!'.format(im1, im2))
        if self.to_rgb:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

        for op in self.transforms:
            outputs = op(im1, im2)
            im1 = outputs[0]
            im2 = outputs[1]
        im1 = np.transpose(im1, (2, 0, 1))
        im2 = np.transpose(im2, (2, 0, 1))
        return im1, im2


# 随机旋转
class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im1, im2):
        if random.random() < self.prob:
            im1 = horizontal_flip(im1)
            im2 = horizontal_flip(im2)
        return im1, im2


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im1, im2):
        if random.random() < self.prob:
            im1 = vertical_flip(im1)
            im2 = vertical_flip(im2)
        return im1, im2


def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=(256, 256), interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        h = target_size[0]
        w = target_size[1]
    else:
        h = target_size
        w = target_size
    # cv2 先宽后高
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


# 归一化
class Normalize:

    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im1, im2):

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im1 = normalize(im1, mean, std)
        im2 = normalize(im2, mean, std)

        return im1, im2


class RandomResize:
    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))

    def __call__(self, im1, im2):

        if not isinstance(im1, np.ndarray) or not (im2, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im1.shape) != 3 or len(im2.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        target_size_list = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        target_size = random.sample(target_size_list, 1)[0]
        target_size = (target_size, target_size)
        im1 = resize(im1, target_size, self.interp_dict[interp])

        im2 = resize(im2, target_size, self.interp_dict[interp])

        return im1, im2


class Resize:
    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                        format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                    .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im1, im2):

        if not isinstance(im1, np.ndarray) or not (im2, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im1.shape) != 3 or len(im2.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im1 = resize(im1, self.target_size, self.interp_dict[interp])

        im2 = resize(im2, self.target_size, self.interp_dict[interp])

        return im1, im2


class SplitIntoParts:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def __call__(self, img):
        height, width, channel = img.shape
        # 将图片缩放到最接近256的整数倍的尺寸
        num_h = height // self.target_size[0]
        num_w = width // self.target_size[1]

        assert num_w > 0 and num_h > 0
        img = resize(img, (num_h * self.target_size[0], num_w * self.target_size[1]))
        img_parts = np.zeros(shape=[num_h, num_w, self.target_size[0], self.target_size[1], channel], dtype=np.float32)
        for i in range(num_h):
            for j in range(num_w):
                img_parts[i, j, :, :, :] = img[i * self.target_size[0]:(i + 1) * self.target_size[0],
                                           j * self.target_size[1]:(j + 1) * self.target_size[1], :]
        return img_parts


class Elastic_Transform:
    def __init__(self, rate=0.5):
        self.choice = random.random()
        self.rate = rate  # 做弹性形变的几率

    def __call__(self, im1, im2):
        if (self.choice < self.rate):
            return im1, im2

        rand_seed = int(time.time())
        im1 = elastic_transform(im1, im1.shape[1] * 2, im1.shape[1] * 0.08, im1.shape[1] * 0.08, random_seed=rand_seed)
        im2 = elastic_transform(im2, im1.shape[1] * 2, im1.shape[1] * 0.08, im1.shape[1] * 0.08, random_seed=rand_seed)
        return im1, im2


def elastic_transform(image, alpha, sigma,
                      alpha_affine, random_seed=None):
    if random_seed is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(seed=random_seed)
    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)
    # generate meshgrid
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
    # bilinear interpolation
    imageC = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return imageC


class MixUp(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, img1, gt1, img2, gt2):
        assert img1.shape == img2.shape
        return self.mixUp(img1.copy(), gt1.copy(), img2.copy(), gt2.copy())

    def mixUp(self, img1, gt1, img2, gt2):
        mix_img = self.ratio * img1
        mix_gt = self.ratio * gt1
        mix_img += (1 - self.ratio) * img2
        mix_gt += (1 - self.ratio) * gt2
        return mix_img, mix_gt
