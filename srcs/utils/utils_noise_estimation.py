#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import torch
from cv2 import imread
from skimage import img_as_float
import time
import sys

# ------------
# # Power by Zongsheng Yue 2019-01-07 14:36:55
# https://github.com/zsyOAOA/noise_est_ICCV2015/blob/master/noise_estimation.py
# numpy version
# --------------


def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))


def noise_estimate(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    else:
        im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii] > tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)

# ------------
# # Power by Zongsheng Yue 2019-01-07 14:36:55
# https://github.com/zsyOAOA/noise_est_ICCV2015/blob/master/noise_estimation.py
# pytorch version, Zhihong Zhang
# --------------


def im2patch_torch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, torch tensor format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = torch.zeros((C, pch_H*pch_W, num_pch)).type_as(im)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))


def noise_estimate_torch(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, N x 3 x H x W  or N x 1 x H x W  torch tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if im.ndim == 3:
        im = torch.unsqueeze(im, 0)
    tau_all = torch.zeros(im.shape[0]).cuda()
    for k in range(im.shape[0]):
        # image to patch
        # C x pch_size x pch_size x num_pch tensor
        pch = im2patch_torch(im[k], pch_size, 3)
        num_pch = pch.shape[3]
        pch = pch.view((-1, num_pch))  # d x num_pch matrix
        d = pch.shape[0]

        mu = pch.mean(axis=1, keepdims=True)  # d x 1
        X = pch - mu
        sigma_X = torch.matmul(X, X.permute(1, 0)) / num_pch
        sig_value, _ = torch.linalg.eigh(sigma_X)
        sig_value = torch.sort(sig_value)[0]

        for ii in range(-1, -d-1, -1):
            tau = torch.mean(sig_value[:ii])
            if torch.sum(sig_value[:ii] > tau) == torch.sum(sig_value[:ii] < tau):
                tau_all[k] = tau
                break
    return torch.sqrt(tau_all)



# main test
if __name__ == '__main__':
    from utils_deblur_dwdn import MedianPool2d
    im = imread('./outputs/testimg/clear01.png')
    im = img_as_float(im)

    # noise_level = [5, 15, 20, 30, 40]
    noise_level = [0.01, 0.03, 0.05, 0.10, 0.20]

    # === numpy version noise estimation ===
    print('===== numpy version ======')
    for level in noise_level:
        # sigma = level / 255
        sigma = level

        im_noise = im + np.random.randn(*im.shape) * sigma

        start = time.time()
        est_level = noise_estimate(im_noise, 8)
        end = time.time()
        time_elapsed = end - start

        str_p = "Time: {0:.4f}, Ture Level: {1:6.4f}, Estimated Level: {2:6.4f}"
        # print(str_p.format(time_elapsed, level, est_level*255))
        print(str_p.format(time_elapsed, level, est_level))

    # === torch version noise estimation ===
    print('===== torch version ======')
    im_tensor = torch.tensor(
        im, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
    for level in noise_level:
        # sigma = level / 255
        sigma = level

        im_noise_tensor = im_tensor + torch.randn(*im_tensor.shape) * sigma

        start = time.time()
        est_level = noise_estimate_torch(im_noise_tensor, 8)
        end = time.time()
        time_elapsed = end - start

        str_p = "Time: {0:.4f}, Ture Level: {1:6.4f}, Estimated Level:"
        # print(str_p.format(time_elapsed, level, est_level*255))
        print(str_p.format(time_elapsed, level), est_level)


