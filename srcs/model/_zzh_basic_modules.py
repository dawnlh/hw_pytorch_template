from torch.autograd import Function
from typing import Any, NewType
from torch.nn.parameter import Parameter
from torch.nn import Module
from scipy.signal import convolve2d, correlate2d
import numpy as np
from numpy import flip
import torch
from torch.autograd import Function as AGF
from torch.autograd import Variable as AGV
import torch.nn as nn
import torch.nn.functional as F

# ===========================================
# Author: Zhihong Zhang, 2021
# https://github.com/dawnlh
# ===========================================


# --------------------------------------------
# Binarized module by Straight-through estimator (STE)
# --------------------------------------------

# ----- Imp 1 Start -----
class LBSign(torch.autograd.Function):
    """Return -1 if x < 0, 1 if x > 0, 0 if x==0."""
    @staticmethod
    def forward(ctx, input):
        result = torch.sign(input)
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output
# ----- Imp 1 End -----

# ----- Imp 2 Start -----
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.


# A type where each element is in {-1, 1}
BinaryTensor = NewType('BinaryTensor', torch.Tensor)


def binary_sign(x: torch.Tensor) -> BinaryTensor:
    """Return -1 if x < 0, 1 if x >= 0."""
    return x.sign() + (x == 0).type(torch.float)  # type: ignore


class STESign(Function):
    """
    Binarize tensor using sign function.
    Straight-Through Estimator (STE) is used to approximate the gradient of sign function.
    See:
    Bengio, Yoshua, Nicholas Léonard, and Aaron Courville.
    "Estimating or propagating gradients through stochastic neurons for
     conditional computation." arXiv preprint arXiv:1308.3432 (2013).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> BinaryTensor:  # type: ignore
        """
        Return a Sign tensor.
        Args:
            ctx: context
            x: input tensor
        Returns:
            Sign(x) = (x>=0) - (x<0)
            Output type is float tensor where each element is either -1 or 1.
        """
        ctx.save_for_backward(x)
        sign_x = binary_sign(x)
        return sign_x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore  # pragma: no cover (since this is called by C++ code) # noqa: E501
        """
        Compute gradient using STE.
        Args:
            ctx: context
            grad_output: gradient w.r.t. output of Sign
        Returns:
            Gradient w.r.t. input of the Sign function
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.gt(1)] = 0
        grad_input[x.lt(-1)] = 0
        return grad_input


# Convenience function to binarize tensors
binarize = STESign.apply    # type: ignore
# ----- Imp 2 End -----

# ----- 0-1 binarize function -----


def binary1(input):
    '''
    binarize input to 0|1
    # method1: use torch.sign
    '''
    return (LBSign.apply(input)+1)/2


def binary2(input, expn=50):
    '''
    binarize input to 0|1
    # method2: use 1/(1+torch.exp(-expn*x)) to approximate
    '''
    return 1/(1+torch.exp(-expn*input))


# --------------------------------------------
# padding and crop with pytorch
# --------------------------------------------

def pad2same_size(x1, x2):
    '''
    pad x1 or x2 to the same size (the size of the larger one)
    '''
    diffX = x2.size()[3] - x1.size()[3]
    diffY = x2.size()[2] - x1.size()[2]

    if diffX == 0 and diffY == 0:
        return x1, x2

    if diffX >= 0 and diffY >= 0:
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))
    elif diffX < 0 and diffY < 0:
        x2 = nn.functional.pad(
            x2, (-diffX // 2, -diffX + diffX//2, -diffY // 2, -diffY + diffY//2))
    elif diffX >= 0 and diffY < 0:
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2, 0, 0))
        x2 = nn.functional.pad(
            x2, (0, 0, -diffY // 2, -diffY + diffY//2))
    elif diffX < 0 and diffY >= 0:
        x1 = nn.functional.pad(x1, (0, 0, diffY // 2, diffY - diffY//2))
        x2 = nn.functional.pad(
            x2, (-diffX // 2, -diffX + diffX//2, 0, 0))

    return x1, x2


def pad2size(x, size):
    '''
    pad x to given size
    x: N*C*H*W
    size: H'*W'
    '''
    diffX = size[1] - x.size()[3]
    diffY = size[0] - x.size()[2]

    if diffX == 0 and diffY == 0:
        return x

    if diffX >= 0 and diffY >= 0:
        x = nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                  diffY // 2, diffY - diffY//2))
    elif diffX < 0 and diffY < 0:
        x = x[:, :, -diffY // 2: diffY + (-diffY) //
              2, -diffX // 2: diffX + (-diffX)//2]
    elif diffX >= 0 and diffY < 0:
        x = x[:, :, -diffY // 2: diffY + (-diffY)//2, :]
        x = nn.functional.pad(x, (diffX // 2, diffX - diffX//2, 0, 0))
    elif diffX < 0 and diffY >= 0:
        x = x[:, :, :, -diffX // 2: diffX + (-diffX)//2]
        x = nn.functional.pad(x, (0, 0, diffY // 2, diffY - diffY//2))
    return x


def circular_padding(tensor, psf_sz):
    '''
    circularly pad tensor to realize convert conv2d 'valid' mode to 'circular' mode
    '''
    x_pad_len, y_pad_len = psf_sz[0]-1, psf_sz[1]-1
    pad_width = (y_pad_len//2, y_pad_len-y_pad_len//2,
                 x_pad_len//2, x_pad_len-x_pad_len//2)
    tensor = F.pad(tensor, pad_width, "circular")
    return tensor

# --------------------------------------------
# real convolution implemented with pytorch
# --------------------------------------------


def torch_real_conv2d(tensor, psf, mode='circular'):
    '''
    tensor:     N*C*H*W, torch.float32 tensor
    psf:        n*C*h*w, torch.float32 tensor
    mode:       conlolution mode, 'circular' | 'same' | 'valid'
    $return:    Nn*C*H'*W', each 2D H*W image will convolve with $n 2D h*w psf 
    Note: this code will cause very large CPU memory if not use GPU/CUDA
    '''
    # padding for 'circular' mode
    x_pad_len, y_pad_len = psf.shape[-2]-1, psf.shape[-1]-1
    if mode == 'circular':
        pad_width = (y_pad_len//2, y_pad_len-y_pad_len//2,
                     x_pad_len//2, x_pad_len-x_pad_len//2)
        tensor = F.pad(tensor, pad_width, "circular")
        mode = 'valid'  # after padding, use valid to conduct convolve

    # flip psf (F.conv2d is corr in fact)
    psf = torch.flip(psf, [-2, -1])

    # reshape
    n, c, h, w = list(psf.shape)
    psf = psf.permute(1, 0, 2, 3).reshape(n*c, 1, h, w)

    # conv
    conv_tensor = F.conv2d(tensor, psf, groups=tensor.shape[1], padding=mode)

    # reshape
    N, C, H, W = list(conv_tensor.shape)
    conv_tensor = conv_tensor.reshape(c, n*N, H, W).permute(1, 0, 2, 3)

    return conv_tensor

# --------------------------------------------
# convolutional layer implemented with scipy
# --------------------------------------------


class ScipyConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        if bias is not None:
            result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height, bias=None):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))

        if bias:
            self.bias = Parameter(torch.randn(1, 1))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)
