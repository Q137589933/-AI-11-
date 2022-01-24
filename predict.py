import argparse
import glob
import os.path
from copy import copy

import numpy
import paddle
import paddle.nn as nn
import cv2
import time

from PIL import Image

from model import *

import os
import warnings

from transforms import horizontal_flip, vertical_flip, resize
from utils import load_pretrained_model

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='data/data121607/data/test')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default='train_result/model/epoch_100/model.pdparams')

    return parser.parse_args()


def predict(img, model, input_size=[256, 256]):
    h_num, w_num = img.shape[0] // input_size[0], img.shape[1] // input_size[1]
    blur_img = np.zeros(shape=img.shape, dtype=np.float32)
    for h in range(h_num):
        for w in range(w_num):
            img_part = paddle.to_tensor(
                img[h * input_size[0]:(h + 1) * input_size[0], w * input_size[1]:(w + 1) * input_size[1], :],
                dtype=paddle.float32)
            img_part = img_part.unsqueeze(0).transpose((0, 3, 1, 2))
            img_out = model(img_part)
            img_out = nn.functional.interpolate(img_out, size=input_size, mode="bilinear")
            img_out = img_out * 255.0
            img_out = paddle.clip(img_out, 0, 255)
            img_out = img_out.squeeze()
            img_out = paddle.transpose(img_out, [1, 2, 0])
            img_out = img_out.numpy()
            blur_img[h * input_size[0]:(h + 1) * input_size[0],
            w * input_size[1]:(w + 1) * input_size[1],
            :] = img_out
    # 边缘
    h_remain = img.shape[0] - h_num * input_size[0]
    w_remain = img.shape[1] - w_num * input_size[1]

    if h_remain != 0:
        # 剩余高度
        for w in range(w_num):
            img_part = paddle.to_tensor(
                img[-input_size[0]:, w * input_size[1]:(w + 1) * input_size[1], :],
                dtype=paddle.float32
            )
            img_part = img_part.unsqueeze(0).transpose((0, 3, 1, 2))
            img_out = model(img_part)
            img_out = nn.functional.interpolate(img_out, size=input_size, mode="bilinear")
            img_out = img_out * 255.0
            img_out = paddle.clip(img_out, 0, 255)
            img_out = img_out.squeeze()
            img_out = paddle.transpose(img_out, [1, 2, 0])
            img_out = img_out.numpy()
            blur_img[-h_remain:, w * input_size[1]:(w + 1) * input_size[1], :] = img_out[-h_remain:, :, :]
    if w_remain != 0:
        # 剩余宽度

        for h in range(h_num):
            img_part = paddle.to_tensor(
                img[h * input_size[0]:(h + 1) * input_size[0], -input_size[1]:, :],
                dtype=paddle.float32
            )
            img_part = img_part.unsqueeze(0).transpose((0, 3, 1, 2))
            img_out = model(img_part)
            img_out = nn.functional.interpolate(img_out, size=input_size, mode="bilinear")
            img_out = img_out * 255.0
            img_out = paddle.clip(img_out, 0, 255)
            img_out = img_out.squeeze()
            img_out = paddle.transpose(img_out, [1, 2, 0])
            img_out = img_out.numpy()
            blur_img[h * input_size[0]:(h + 1) * input_size[0], -w_remain:, :] = img_out[:, -w_remain:, :]
    if w_remain != 0 and h_remain != 0:
        img_part = paddle.to_tensor(
            img[-input_size[0]:, -input_size[1]:, :],
            dtype=paddle.float32
        )
        img_part = img_part.unsqueeze(0).transpose((0, 3, 1, 2))
        img_out = model(img_part)
        img_out = nn.functional.interpolate(img_out, size=input_size, mode="bilinear")
        img_out = img_out * 255.0
        img_out = paddle.clip(img_out, 0, 255)
        img_out = img_out.squeeze()
        img_out = paddle.transpose(img_out, [1, 2, 0])
        img_out = img_out.numpy()
        blur_img[-h_remain:, -w_remain:, :] = img_out[-h_remain:, -w_remain:, :]
    return blur_img


def main(args):
    multi_output = False
    model = MBCNN(64, multi_output)  # MBCNN-light: model = MBCNN(32,multi_output)
    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)
    im_files = glob.glob(os.path.join(args.dataset_root, "images/*.jpg"))
    input_size = [256, 256]
    for i, im in enumerate(im_files):
        start = time.time()
        img = cv2.imread(im)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_h,ori_w=img.shape[0],img.shape[1]
        if ori_h > 1500 or ori_w > 1500:
            img = resize(img, (ori_h *2//3, ori_w *2//3))
        model.eval()
        img = paddle.to_tensor(img)
        img /= 255.0
        blur_img = np.zeros(shape=img.shape, dtype=np.float32)
        blur_img += 0.25 * predict(img, model, input_size)
        h_img = horizontal_flip(img)
        blur_img += 0.25 * predict(h_img, model, input_size)
        vh_img = vertical_flip(h_img)
        blur_img += 0.25 * horizontal_flip(predict(vh_img, model, input_size))
        v_img = horizontal_flip(vh_img)
        blur_img += 0.25 * horizontal_flip(predict(v_img, model, input_size))
        blur_img=resize(blur_img,(ori_h,ori_w))
        blur_img = np.clip(blur_img, 0, 255)
        save_path = "output/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, im.split('/')[-1]), blur_img)

        end = time.time()
        time_one = end - start
        print('The running time of an image is : {:2f} s'.format(time_one))


if __name__ == '__main__':
    args = parse_args()
    main(args)
