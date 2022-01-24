import glob
import os, cv2, random
import numpy as np
from math import log10
from datetime import datetime
import time

import paddle.nn
from PIL import Image
from paddle.io import Dataset
from paddle.io import DataLoader

from paddle.vision import image_load, set_image_backend
from random import sample
from PIL import Image, ImageFile
import os
from tqdm import tqdm

# the image dir of testing input
from transforms import RandomResize, Compose, RandomHorizontalFlip, RandomCrop, Normalize, Resize, Elastic_Transform, \
    CropCenter, RandomVerticalFlip, ColorJitter

test_path = '/home/aistudio/data/testA/moire_testA_dataset/images'
# the image dir of validation groundtruth
gt_path = '/home/aistudio/data/train/moire_train_dataset/images'
# the image dir of validation input
ns_path = '/home/aistudio/data/train/moire_train_dataset/gts'


class ImageDataset(Dataset):
    def __init__(self, root=None, training=True, val=False, use_elastic=False, crop_center=False):
        super().__init__()
        if root is None:
            raise ValueError("dataset_root is None")
        self.dataset_root = root

        self.im1_list = glob.glob(os.path.join(self.dataset_root, "images", "*.*"))
        self.im1_list.sort()
        self.training = training
        self.val = val
        self.use_elastic = use_elastic
        self.crop_center = crop_center
        if training:
            self.im2_list = glob.glob(os.path.join(self.dataset_root, "gts", "*.*"))
            self.im2_list.sort()
            assert len(self.im1_list) == len(self.im2_list)

    def __getitem__(self, index):
        tf_list = []
        if self.crop_center:
            tf_list.append(CropCenter(rate=0.9))
        if self.training and not self.val:  # 训练集
            # tf_list.append(Resize((256, 256)))
            tf_list.append(RandomCrop((256,256)))
            tf_list.append(RandomHorizontalFlip())
            tf_list.append(RandomVerticalFlip())
            # tf_list.append(ColorJitter())
            tf_list.append(Normalize())
            if self.use_elastic:
                tf_list.append(Elastic_Transform(rate=0.5))
        elif self.training and self.val:  # 验证集
            # tf_list.append(Resize((256, 256)))
            tf_list.append(RandomCrop((256, 256)))
            tf_list.append(Normalize())
        else:  # 测试集
            tf = paddle.vision.transforms.Compose([
                paddle.vision.transforms.Resize((256, 256)),
                paddle.vision.transforms.Normalize()
            ])
        img1 = self.im1_list[index]

        if self.training:
            tf = Compose(tf_list)
            img2 = self.im2_list[index]
            img1, img2 = tf(img1, img2)
            h, w = img1.shape[-2:]
            img2_2 = paddle.vision.transforms.Resize((int(h / 2), int(w / 2)))(img2.transpose((1, 2, 0))).transpose(
                (2, 0, 1))
            img2_3 = paddle.vision.transforms.Resize((int(h / 4), int(w / 4)))(img2.transpose((1, 2, 0))).transpose(
                (2, 0, 1))
            # img2_4 =paddle.vision.transforms.Resize((int(h / 8), int(w / 8)))(img2.transpose((1, 2, 0))).transpose(
            #     (2, 0, 1))
            img2_list = [img2, img2_2, img2_3]
            return img1, img2_list  # 返回多尺度gt，用于计算multi output的loss
        return tf(img1)

    def __len__(self):
        return len(self.im1_list)
