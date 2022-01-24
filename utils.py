import os

import cv2
import paddle
import numpy as np
import scipy.signal
from skimage.metrics import peak_signal_noise_ratio

def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        print('Loading pretrained model from {}'.format(pretrained_model))

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    print("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    print(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                            .format(k, para_state_dict[k].shape,
                                    model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            print("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict),
                model.__class__.__name__))

        else:
            raise ValueError(
                'The pretrained model directory is not Found: {}'.format(
                    pretrained_model))
    else:
        print(
            'No pretrained model to load, {} will be trained from scratch.'.
                format(model.__class__.__name__))


def ssim_index_new(img1, img2, K, win):
    M, N = img1.shape

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (K[0] * 255) ** 2
    C2 = (K[1] * 255) ** 2
    win = win / np.sum(win)

    mu1 = scipy.signal.convolve2d(img1, win, mode='valid')
    mu2 = scipy.signal.convolve2d(img2, win, mode='valid')
    mu1_sq = np.multiply(mu1, mu1)
    mu2_sq = np.multiply(mu2, mu2)
    mu1_mu2 = np.multiply(mu1, mu2)
    sigma1_sq = scipy.signal.convolve2d(np.multiply(img1, img1), win, mode='valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve2d(np.multiply(img2, img2), win, mode='valid') - mu2_sq
    img12 = np.multiply(img1, img2)
    sigma12 = scipy.signal.convolve2d(np.multiply(img1, img2), win, mode='valid') - mu1_mu2

    if (C1 > 0 and C2 > 0):
        ssim1 = 2 * sigma12 + C2
        ssim_map = np.divide(np.multiply((2 * mu1_mu2 + C1), (2 * sigma12 + C2)),
                             np.multiply((mu1_sq + mu2_sq + C1), (sigma1_sq + sigma2_sq + C2)))
        cs_map = np.divide((2 * sigma12 + C2), (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones(mu1.shape)
        index = np.multiply(denominator1, denominator2)
        # 如果index是真，就赋值，是假就原值
        n, m = mu1.shape
        for i in range(n):
            for j in range(m):
                if (index[i][j] > 0):
                    ssim_map[i][j] = numerator1[i][j] * numerator2[i][j] / denominator1[i][j] * denominator2[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]
        for i in range(n):
            for j in range(m):
                if ((denominator1[i][j] != 0) and (denominator2[i][j] == 0)):
                    ssim_map[i][j] = numerator1[i][j] / denominator1[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]

        cs_map = np.ones(mu1.shape)
        for i in range(n):
            for j in range(m):
                if (denominator2[i][j] > 0):
                    cs_map[i][j] = numerator2[i][j] / denominator2[i][j]
                else:
                    cs_map[i][j] = cs_map[i][j]

    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)

    return mssim, mcs


def msssim(img1, img2):
    K = [0.01, 0.03]
    win = np.multiply(cv2.getGaussianKernel(11, 1.5), (cv2.getGaussianKernel(11, 1.5)).T)  # H.shape == (r, c)
    level = 5
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    method = 'product'

    _,M, N = img1.shape
    H, W = win.shape

    downsample_filter = np.ones((2, 2)) / 4
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mssim_array = []
    mcs_array = []

    for i in range(0, level):
        mssim, mcs = ssim_index_new(img1, img2, K, win)
        mssim_array.append(mssim)
        mcs_array.append(mcs)
        filtered_im1 = cv2.filter2D(img1, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
        filtered_im2 = cv2.filter2D(img2, -1, downsample_filter, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
        img1 = filtered_im1[::2, ::2]
        img2 = filtered_im2[::2, ::2]

    print(np.power(mcs_array[:level - 1], weight[:level - 1]))
    print(mssim_array[level - 1] ** weight[level - 1])
    overall_mssim = np.prod(np.power(mcs_array[:level - 1], weight[:level - 1])) * (
            mssim_array[level - 1] ** weight[level - 1])

    return overall_mssim


def psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    # mse = ((im1.astype(np.double) - im2.astype(np.double)) ** 2).mean()
    # psnr = 10 * np.log10(255.0 ** 2 / mse)
    return peak_signal_noise_ratio(np.array(im1),np.array(im2))


def cal_score(output, gt):
    PSNR = psnr(output.numpy(), gt.numpy())
    MSSSIM = msssim(output.numpy(), gt.numpy())
    score = 0.5 * PSNR + 0.5 * MSSSIM
    return PSNR, MSSSIM, score


def split_into_parts(img, target_size=(480, 480)):
    _, width, height = img.shape
    sum_w = width / target_size[0] + 1  # 横方向划分数
    sum_h = height / target_size[1] + 1

