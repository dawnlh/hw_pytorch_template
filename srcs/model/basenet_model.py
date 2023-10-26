
import torch.nn as nn
from srcs.model._zzh_denoiser_modules import ResUNet

# ===========================
# A basic denoising neural network (ResUNet) for project demonstration
# ===========================


class basenet(nn.Module):
    '''
    image and feature space multi-scale deconvolution network, cross residual fusion:
    '''

    def __init__(self, n_colors, nc):
        super(basenet, self).__init__()
        # network architecture
        self.ResUNet = ResUNet(in_nc=n_colors, out_nc=n_colors, nc=nc)

    def forward(self, img):
        ## forward process
        out = self.ResUNet(img)

        ## return
        return out
