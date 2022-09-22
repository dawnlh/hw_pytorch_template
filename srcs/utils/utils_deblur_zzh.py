# -*- coding: utf-8 -*-
import numpy as np
import scipy
import cv2
from math import cos, sin
from numpy import zeros, ones, prod, array, pi, log, maximum, mod, arange, sum, mgrid, exp, pad, round, ceil, floor
from numpy.random import randn, rand, randint, uniform
from scipy.signal import convolve2d
import torch
import os
import random
from os.path import join as opj
import torch.nn.functional as F
from scipy.signal import fftconvolve

'''
some codes are copied/modified from
    https://github.com/cszn
    https://github.com/twhui/SRGAN-pyTorch
    https://github.com/xinntao/BasicSR
    https://gitlab.mpi-klsb.mpg.de/jdong/dwdn

Last modified: 2021-10-28 Zhihong Zhang
'''


# ===============
# Fourier transformation
# ===============


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        # otf: NxCxHxWx2
        otf: NxCxHxW
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fft2(otf, dim=(-2, -1))
    return otf

# ===============
# circular padding
# ===============



def pad_circular(x, pad):
    """
    2D image circular padding: pad each side of x with pad elements
    :param x: img, shape [H, W]
    :param pad: padding size, int >= 0
    :return: padded res, [H+2*pad, W+2*pad]
    """
    x = torch.cat([x, x[0:pad]], dim=0)
    x = torch.cat([x, x[:, 0:pad]], dim=1)
    x = torch.cat([x[-2 * pad:-pad], x], dim=0)
    x = torch.cat([x[:, -2 * pad:-pad], x], dim=1)

    return x



def pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: padding size, int >= 0
    :param dim: the dimension over which the tensors are padded
    :return: padded res
    """

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")

        idx = tuple(slice(0, None if s != d else pad, 1)
                    for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)

        idx = tuple(slice(None if s != d else -2 * pad, None if s !=
                          d else -pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x[idx], x], dim=d)
        pass

    return x


# ===============
# edge taper
# ===============
# Implementation from https://github.com/teboli/polyblur


def edgetaper(img, kernel, n_tapers=3):
    if type(img) == np.ndarray:
        return edgetaper_np(img, kernel, n_tapers)
    else:
        return edgetaper_torch(img, kernel, n_tapers)


def pad_for_kernel_np(img, kernel, mode):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p, p] + (img.ndim-2)*[(0, 0)]
    return np.pad(img, padding, mode)


def crop_for_kernel_np(img, kernel):
    p = [(d-1)//2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim-2) * [slice(None)]
    return img[r]


def edgetaper_alpha_np(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1-i), img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z / np.max(z))
    return np.outer(*v)


def edgetaper_np(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha_np(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel_np(
            img, _kernel, 'wrap'), kernel, mode='valid')
        img = alpha*img + (1-alpha)*blurred
    return img


def edgetaper_alpha_torch(kernel, img_shape):
    z = torch.fft.fft(torch.sum(kernel, -1), img_shape[0]-1)
    z = torch.real(torch.fft.ifft(torch.abs(z)**2)).float()
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v1 = 1 - z / torch.max(z)

    z = torch.fft.fft(torch.sum(kernel, -2), img_shape[1] - 1)
    z = torch.real(torch.fft.ifft(torch.abs(z) ** 2)).float()
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v2 = 1 - z / torch.max(z)

    return v1.unsqueeze(-1) * v2.unsqueeze(-2)


def edgetaper_torch(img, kernel, n_tapers=3):
    h, w = img.shape[-2:]
    alpha = edgetaper_alpha_torch(kernel, (h, w))
    _kernel = kernel
    ks = _kernel.shape[-1] // 2
    for i in range(n_tapers):
        img_padded = F.pad(img, [ks, ks, ks, ks], mode='circular')
        K = p2o(kernel, img_padded.shape[-2:])
        I = torch.fft.fft2(img_padded)
        blurred = torch.real(torch.fft.ifft2(K * I))[..., ks:-ks, ks:-ks]
        img = alpha * img + (1 - alpha) * blurred
    return img


if __name__ == '__main__':
    pass
