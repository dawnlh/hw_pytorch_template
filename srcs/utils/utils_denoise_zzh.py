import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

'''
Zhihong Zhang, 2022-03-14
Reference:
    https://github.com/cszn
    https://github.com/dongjxjx/dwdn

'''


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(
            3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def image_nsr(_input_blur):
    # image noise-signal-ratio (amplitude) N*C
    # ref: DWDN/wiener_filter_para
    median_filter = MedianPool2d(
        kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = (diff.shape[2]*diff.shape[3])
    mean_n = torch.sum(diff, (2, 3), keepdim=True).repeat(
        1, 1, diff.shape[2], diff.shape[3])/num
    var_n2 = (torch.sum((diff - mean_n) * (diff - mean_n),
                        (2, 3))/(num-1))**(0.5)

    var_s2 = (torch.sum((median_filter) * (median_filter), (2, 3))/(num-1))**(0.5)
    # NSR = var_n2 / var_s2 * 8.0 / 3.0 / 10.0
    NSR = var_n2 / var_s2

    return NSR


# main test
if __name__ == '__main__':
    import cv2
    import skimage
    import time
    im = cv2.imread('./outputs/testimg/clear01.png')
    im = skimage.img_as_float(im)

    # noise_level = [5, 15, 20, 30, 40]
    noise_level = [0.01, 0.03, 0.05, 0.10, 0.20]
    # === torch version nsr ===
    im_tensor = torch.tensor(
        im, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
    for level in noise_level:
        # sigma = level / 255
        sigma = level

        im_noise_tensor = im_tensor + torch.randn(*im_tensor.shape) * sigma

        start = time.time()
        nsr = image_nsr(im_noise_tensor).repeat(
            1, 1, im_noise_tensor.shape[2], im_noise_tensor.shape[3])

        end = time.time()
        time_elapsed = end - start

        str_p = "Time: {0:.4f}, Ture Level: {1:6.4f}, Estimated Level:"
        # print(str_p.format(time_elapsed, level, est_level*255))
        print(str_p.format(time_elapsed, level), nsr.shape)
