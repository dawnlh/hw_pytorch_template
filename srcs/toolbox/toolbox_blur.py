# -*- coding: utf-8 -*-
from functools import reduce
import numpy as np
import scipy
from scipy import fftpack, ndimage
import os
import sys
from math import cos, sin
from numpy import zeros, ones, prod, array, pi, log, min, max, maximum, mod, arange, sum, mgrid, exp, pad, round, ceil, floor
from numpy.random import randn, rand, randint, uniform
from scipy.signal import convolve2d
import cv2
from os.path import join as opj
import scipy.io as sio


'''
some codes are copied/modified from 
    Kai Zhang (github: https://github.com/cszn)
    https://github.com/twhui/SRGAN-pyTorch
    https://github.com/xinntao/BasicSR

Last modified: 2021-10-28 Zhihong Zhang 
'''

# ===============
# blur kernels generation
# ===============

# ----- fspecial -----


def fspecial_gauss(size, sigma):
    x, y = mgrid[-size // 2 + 1: size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    g = exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1),
                         np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h

# ----- (coded) Linear Motion Blur -----


def linearMotionBlurKernel(motion_len=[15, 35], theta=[0, 2*pi], psf_sz=50):
    '''
    linear motion blur kernel
    kernel_len: kernel length range (pixel)
    theta: motion direction range (rad)
    psf_sz: psf size (pixel)
    '''
    if not isinstance(psf_sz, list):
        psf_sz = [psf_sz, psf_sz]
    if not isinstance(motion_len, list):
        motion_len = [motion_len, motion_len]
    if not isinstance(theta, list):
        theta = [theta, theta]

    motion_len = uniform(*motion_len)
    theta = uniform(*theta)

    motion_len_n = ceil(motion_len*2).astype(int)  # num of points

    # get random trajectory
    try_times = 0
    while(True):
        x = zeros((2, motion_len_n))
        Lx = motion_len*cos(theta)
        Ly = motion_len*sin(theta)
        x[0] = round(np.linspace(0, Lx, motion_len_n))
        x[1] = round(np.linspace(0, Ly, motion_len_n))
        x = x.astype(int)
        # traj threshold judge
        x[0], x[1] = x[0]-min(x[0]), x[1]-min(x[1])  # move to first quadrant
        x_thr = [max(x[0])+1, max(x[1]+1)]
        if ((np.array(psf_sz) - np.array(x_thr)) > 0).all():
            break  # proper trajectory with length < psf_size

        try_times = try_times+1
        assert try_times < 10, 'Error: MOTION_LEN > PSF_SZ'

    # get kernel
    k = zeros(x_thr)
    for x_i in x.T:
        k[x_i[0], x_i[1]] = 1

    # padding
    pad_width = (psf_sz[0] - x_thr[0], psf_sz[1] - x_thr[1])
    pad_width = ((pad_width[0]//2, pad_width[0]-pad_width[0]//2),
                 (pad_width[1]//2, pad_width[1]-pad_width[1]//2))

    assert (np.array(pad_width) > 0).all(), 'Error: MOTION_LEN > PSF_SZ'
    k = pad(k, pad_width, 'constant')

    # guassian blur
    k = np.rot90(k, -1)
    k = k/sum(k)
    k = convolve2d(k, fspecial_gauss(2, 1), "same")  # gaussian blur
    k = k/sum(k)

    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    return k


def codedLinearMotionBlurKernel(motion_len=[15, 35], theta=[0, 2*pi], psf_sz=50, code=None):
    '''
    linear motion blur kernel
    kernel_len: kernel length range (pixel)
    theta: motion direction range (rad)
    psf_sz: psf size (pixel)
    code: flutter shutter code
    '''
    if isinstance(psf_sz, int):
        psf_sz = [psf_sz, psf_sz]
    if isinstance(motion_len, (int, float)):
        motion_len = [motion_len, motion_len]
    if isinstance(theta, (int, float)):
        theta = [theta, theta]

    motion_len = uniform(*motion_len)
    theta = uniform(*theta)

    # get coded trajectory
    # code matching to motion length
    if code is None:
        motion_len_n = ceil(motion_len*3).astype(int)  # num of points
        code_n = ones((1, motion_len_n))
    else:
        motion_len_n = ceil(maximum(motion_len, len(code))*3).astype(int)
        code_n = [code[floor(k*len(code)/motion_len_n).astype(int)]
                  for k in range(motion_len_n)]
    # print(code, '\n', code_n)

    # get random trajectory
    try_times = 0
    while(True):
        x = zeros((2, motion_len_n))
        Lx = motion_len*cos(theta)
        Ly = motion_len*sin(theta)
        x[0] = round(np.linspace(0, Lx, motion_len_n))
        x[1] = round(np.linspace(0, Ly, motion_len_n))
        x = x*code_n  # coded traj
        x = x.astype(int)
        # traj threshold judge
        x[0], x[1] = x[0]-min(x[0]), x[1]-min(x[1])  # move to first quadrant
        x_thr = [max(x[0])+1, max(x[1]+1)]
        if ((np.array(psf_sz) - np.array(x_thr)) > 0).all():
            break  # proper trajectory with length < psf_size

        try_times = try_times+1
        assert try_times < 10, 'Error: MOTION_LEN > PSF_SZ'

    # get kernel
    k = zeros(x_thr)
    for x_i in x.T:
        k[x_i[0], x_i[1]] = 1

    # padding
    pad_width = (psf_sz[0] - x_thr[0], psf_sz[1] - x_thr[1])
    pad_width = ((pad_width[0]//2, pad_width[0]-pad_width[0]//2),
                 (pad_width[1]//2, pad_width[1]-pad_width[1]//2))

    k = pad(k, pad_width, 'constant')

    # guassian blur
    k = np.rot90(k, -1)
    k = k/sum(k)
    k = convolve2d(k, fspecial_gauss(2, 1), "same")  # gaussian blur
    k = k/sum(k)

    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    return k


def linearTrajectory(T_value):
    '''
    get a linear trajectory coordinate sequence with length of T pixels (or length belong to [T[0],T[1]])
    '''
    if isinstance(T_value, (int, float)):
        T_value = T_value
    else:
        T_value = randint(T_value[0], T_value[1])

    # original point = [0,0], direction = theta
    theta = rand()*2*pi
    Tx = T_value*cos(theta)
    Ty = T_value*sin(theta)

    x = zeros((2, T_value))
    x[0] = np.linspace(0, Tx, T_value)
    x[1] = np.linspace(0, Ty, T_value)

    return x


# ----- Random Motion Blur -----

def codedRandomMotionBlurKernelPair(motion_len_r=[15, 35],  psf_sz=50, code=None):
    '''
    a pair of coded and non-coded random motion blur kernel
    kernel_len: kernel length range (pixel)
    psf_sz: psf size (pixel)
    code: flutter shutter code
    '''
    if isinstance(psf_sz, int):
        psf_sz = [psf_sz, psf_sz]
    if isinstance(motion_len_r, (int, float)):
        motion_len_r = [motion_len_r, motion_len_r]
    # if isinstance(theta, (int, float)):
    #     theta = [theta, theta]

    motion_len_v = uniform(*motion_len_r)

    # get coded trajectory
    # code matching to motion length
    if code is None:
        # num of sampling points
        motion_len_n = ceil(motion_len_v*3).astype(int)
        code_n = ones((1, motion_len_n))
    else:
        code = np.array(code, dtype=np.float32)
        motion_len_n = ceil(maximum(motion_len_v, len(code))*3).astype(int)
        code_n = [code[floor(k*len(code)/motion_len_n).astype(int)]
                  for k in range(motion_len_n)]
    # print(code, '\n', code_n)

    # get random trajectory
    x = getRandomTrajectory(motion_len_r, motion_len_n,
                            psf_sz, len_param=motion_len_v, curve_param=4)
    # x = x*code_n  # coded traj

    k = traj2kernel(x, psf_sz, traj_v=code_n)
    k = convolve2d(k, fspecial_gauss(2, 1), "same")  # gaussian blur
    k = k/sum(k)

    k_orig = traj2kernel(x, psf_sz, traj_v=1)
    k_orig = convolve2d(k_orig, fspecial_gauss(2, 1), "same")  # gaussian blur
    k_orig = k_orig/sum(k_orig)
    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    return k, k_orig


def traj2kernel(x, psf_sz, traj_v=1):
    '''
    convert trajectory to blur kernel
    x: traj coord, 2*N
    psf_sz: psf size
    traj_v: value of trajectory points, scalar | 1*N list
    '''

    if isinstance(psf_sz, int):
        psf_sz = [psf_sz, psf_sz]
    if isinstance(traj_v, (int, float)):
        traj_v = [traj_v]*x.shape[1]
    traj_v = np.array(traj_v).squeeze()

    x = x.astype(int)
    x[0], x[1] = x[0]-min(x[0]), x[1]-min(x[1])  # move to first quadrant
    x_thr = [max(x[0])+1, max(x[1]+1)]
    psf = zeros(x_thr)
    for n, x_i in enumerate(x.T):
        psf[x_i[0], x_i[1]] += traj_v[n]

    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    # padding
    pad_width = (psf_sz[0] - x_thr[0], psf_sz[1] - x_thr[1])
    pad_width = ((pad_width[0]//2, pad_width[0]-pad_width[0]//2),
                 (pad_width[1]//2, pad_width[1]-pad_width[1]//2))

    # assert (np.array(pad_width) > 0).all(), 'Error: MOTION_LEN > PSF_SZ'
    psf = pad(psf, pad_width, 'constant')

    # normalize
    # k = np.rot90(k, -1)
    psf = psf/sum(np.array(traj_v))

    return psf


def getRandomTrajectory(motion_len_r, motion_len_n, motion_thr, len_param=None, curve_param=4, max_try_times=50):
    '''
    generate random traj (MOTION_LEN_N points) with length belong to MOTION_LEN and range within MOTION_THR
    '''
    if isinstance(motion_thr, int):
        motion_thr = [motion_thr, motion_thr]
    if isinstance(motion_len_r, (int, float)):
        motion_len_r = [motion_len_r, motion_len_r]
    if len_param is None:
        len_param = np.random.uniform(*motion_len_r)

    try_times = 0
    while(True):
        x = zeros((2, motion_len_n))
        v = zeros((2, motion_len_n))
        r = zeros((2, motion_len_n))
        trans_delta = len_param/motion_len_n
        rot_delta = 2 * pi / motion_len_n

        for t in range(1, motion_len_n):
            trans_n = randn(2)/(t+1)
            rot_n = randn(2)*curve_param
            # Keep the inertia of volecity
            v[:, t] = v[:, t - 1] + trans_delta * trans_n
            # Keep the inertia of direction
            r[:, t] = r[:, t - 1] + rot_delta * rot_n
            st = rot2D(v[:, t], r[:, t])
            x[:, t] = x[:, t - 1] + st

        # calc trajectory threshold and judge
        x[0], x[1] = x[0]-min(x[0]), x[1]-min(x[1])  # move to first quadrant
        x_thr = [max(x[0])+1, max(x[1]+1)]
        x_len = np.sum(np.array([np.sqrt(np.sum((x[:, k+1]-x[:, k])**2))
                                 for k in range(motion_len_n-1)]))
        if motion_thr[0] > x_thr[0] and motion_thr[1] > x_thr[1] and motion_len_r[0] < x_len < motion_len_r[1]:
            break  # proper trajectory with length < psf_size and length in MOTION_LEN range
        try_times = try_times+1
        assert try_times < max_try_times, 'Error: MOTION_LEN and PSF_SZ is not proper'
    return x


def rot2D(x, r):
    Rx = array([[cos(r[0]), -sin(r[0])],
               [sin(r[0]), cos(r[0])]])
    Ry = array([[cos(r[1]), sin(r[1])],
                [-sin(r[1]), cos(r[1])]])
    R = Ry @ Rx
    x = R @ x
    return x


def psf_blur_img(img, psf, noise_level=0):
    """
    coded exposure psf blurred image

    Args:
        img (ndarray): sharp image
        psf (ndarray): coded exposure psf
        noise_level (scalar): noise level

    Returns:
        x: [description]
    """
    coded_blur_img = ndimage.filters.convolve(
        img, np.expand_dims(psf, axis=2), mode='wrap')
    # add Gaussian noise
    coded_blur_img = coded_blur_img + \
        np.random.normal(0, noise_level, coded_blur_img.shape)
    return coded_blur_img.astype(np.float32)


#=============
# main function
#=============

if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__) + os.sep + '../')
    import matplotlib.pyplot as plt
    from utils import utils_deblur_kair
    from utils.utils_image_zzh import augment_img

# %% Funciton list
    FLAG_generate_psf_pair = False
    FLAG_psf_blur_image = False
    FLAG_traj_psf = False
    FLAG_kair_psf = True


# %% generate random motion trajectory and corresponding psf / load traj and generate psf
if FLAG_traj_psf:
    # params
    motion_len_r = [60, 68]
    motion_len_n = 192
    psf_sz = 80
    iter = 20
    load_traj = False
    traj_dir = './dataset/benchmark/pair_traj_psf1/traj/'
    save_dir = './outputs/tmp/traj/'

    # load traj
    # generate dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir+'box_psf', exist_ok=True)
        os.makedirs(save_dir+'traj', exist_ok=True)

    # run
    for k in range(iter):
        if load_traj:
            traj = sio.loadmat(traj_dir+'traj%02d.mat' % (k+1))
            traj = traj['traj']
        else:
            traj = getRandomTrajectory(motion_len_r, motion_len_n, motion_thr=psf_sz,
                                       len_param=None, curve_param=6, max_try_times=100).astype(np.int32)

        psf = traj2kernel(traj, psf_sz)
        psf = convolve2d(psf, fspecial_gauss(3, 1), "same")  # gaussian blur
        psf = psf/sum(psf)
        psf_png = psf/np.max(psf)*255
        # psf_png = psf_png[:, ::-1]

        print('PSF Num.', k)
        sio.savemat(opj(save_dir, 'traj/traj%02d.mat' % (k+1)), {'traj': traj})
        cv2.imwrite(opj(save_dir, 'box_psf/box_psf%02d.png'
                    % (k+1)), psf_png, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# %% generate coded and non-coded psf pair
if FLAG_generate_psf_pair:
    # param
    motion_len = [25, 48]

    # ce_code = [1, 0, 1, 0, 1]
    # ce_code = [1]*32
    # ce_code = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    # ce_code = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # ce_code = [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    #            1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1]  # [cui2020MultiframeMotion]
    ce_code = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0,
               0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1]  # [raskar2006CodedExposure]
    save_dir = './outputs/tmp/psf/'
    psf_num = 10

    # generate dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir+'ce_psf', exist_ok=True)
        os.makedirs(save_dir+'box_psf', exist_ok=True)

    for k in range(psf_num):
        coded_psf, psf = codedRandomMotionBlurKernelPair(
            motion_len_r=motion_len,  psf_sz=80, code=ce_code)

        psf_png = psf/np.max(psf)*255
        coded_psf_png = coded_psf/np.max(coded_psf)*255

        # import matplotlib.pyplot as plt
        # plt.imshow(psf, interpolation="nearest", cmap="gray")
        # plt.show()

        print('PSF Num.', k)
        cv2.imwrite(opj(save_dir, 'ce_psf/ce_psf%02d.png' %
                    k), coded_psf_png, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(opj(save_dir, 'box_psf/box_psf%02d.png' %
                    k), psf_png, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# %% psf blur image
if FLAG_psf_blur_image:
    # param & path
    noise_level = 0
    img_idxs = list(range(40))  # corresponding idx
    psf_idxs = list(range(40))  # corresponding idx

    psf_dir = '/hdd/1/zzh/project/CED-Net/dataset/benchmark/pair_ce_random_psf1'
    img_dir = '/hdd/1/zzh/project/CED-Net/dataset/benchmark/DIV2K_valid_HR40'
    save_dir = './outputs/tmp/blurImage/'

    # get path & gen dir
    img_names = sorted(os.listdir(img_dir))
    psf_names = sorted(os.listdir(psf_dir))
    img_num = len(img_names)
    psf_num = len(psf_names)

    assert img_num >= len(img_idxs) and psf_num >= len(
        psf_idxs), 'Error: Given idx > Total file num'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # blur image
    cnt = 0
    for psf, t in zip(psf_idxs, img_idxs):
        print('--> PSF-%d, Image-%d' % (psf, t))
        # psf
        psf_k = cv2.imread(opj(psf_dir, psf_names[psf]))
        assert psf_k is not None, 'psf_%d is None' % psf
        psf_k = cv2.cvtColor(psf_k, cv2.COLOR_BGR2GRAY)
        psf_k = psf_k.astype(np.float32)/np.sum(psf_k)

        # image
        img_t = cv2.imread(opj(img_dir, img_names[t]))
        assert img_t is not None, 'img_%d is None' % t
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        img_t = img_t.astype(np.float32)/255

        # blur & noise
        blur_tmp = psf_blur_img(img_t, psf_k)
        if noise_level == 0:
            noisy_blur_tmp = blur_tmp
        else:
            noisy_blur_tmp = blur_tmp + \
                np.random.normal(0, noise_level, blur_tmp.shape)

        noisy_blur_tmp = noisy_blur_tmp[..., ::-1]*255

        cnt += 1
        cv2.imwrite(opj(save_dir, 'blur_img%03d.png' % cnt),
                    noisy_blur_tmp, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# %% generate psf
if FLAG_kair_psf:
    from utils import utils_deblur_kair
    save_dir = './outputs/tmp3/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for k in range(98):
        psf = utils_deblur_kair.blurkernel_synthesis_zzh(37)
        psf = psf/np.max(psf)*255
        print("PSF_%02d" % (k+1))
        cv2.imwrite(opj(save_dir, 'psf%02d.png' % (k+1)),
                    psf, [cv2.IMWRITE_PNG_COMPRESSION, 0])
