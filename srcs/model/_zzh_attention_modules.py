from collections import OrderedDict
from torch.nn import init
from torch import nn
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------
# CSMAttention modules (channel spatial modulation attention)
# --------------------------------------------

class CMAttention(nn.Module):
    def __init__(self, in_planes1, in_planes2):
        super(CMAttention, self).__init__()
        in_planes = in_planes1 + in_planes2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.full_conv_x1 = nn.Conv2d(
        #     in_planes, in_planes, img_sz, groups=in_planes)
        # self.full_conv_x2 = nn.Conv2d(
        #     in_planes, in_planes, img_sz, groups=in_planes)
        # self.full_conv_y1 = nn.Conv2d(
        #     in_planes, in_planes, img_sz, groups=in_planes)
        # self.full_conv_y2 = nn.Conv2d(
        #     in_planes, in_planes, img_sz, groups=in_planes)

        # self.fc1 = nn.Conv2d(8*in_planes, 4*in_planes, 1)
        self.fc1 = nn.Conv2d(2*in_planes, 2*in_planes, 1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(2*in_planes, in_planes, 1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Conv2d(in_planes, in_planes, 1)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Conv2d(in_planes, 2*in_planes, 1)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Conv2d(2*in_planes, 2*in_planes, 1)
        self.sigmoid5 = nn.Sigmoid()

    def forward(self, x, y):
        avg_pool_x = self.avg_pool(x)
        max_pool_x = self.max_pool(x)
        # full_conv_x1 = self.full_conv_x1(x)
        # full_conv_x2 = self.full_conv_x2(x)

        avg_pool_y = self.avg_pool(y)
        max_pool_y = self.max_pool(y)
        # full_conv_y1 = self.full_conv_y1(y)
        # full_conv_y2 = self.full_conv_y2(y)

        # cat_xy = torch.cat((avg_pool_x, max_pool_x,
        #                    full_conv_x1, full_conv_x2, avg_pool_y, max_pool_y, full_conv_y1, full_conv_y2), 1)  # c8
        cat_xy = torch.cat(
            (avg_pool_x, max_pool_x, avg_pool_y, max_pool_y), 1)  # c4

        cat_xy1 = self.relu1(self.fc1(cat_xy))  # c4
        cat_xy2 = self.relu2(self.fc2(cat_xy1))  # c2
        cat_xy3 = self.relu3(self.fc3(cat_xy2))  # c2
        cat_xy4 = self.relu4(self.fc4(cat_xy3))  # c4
        cat_xy5 = self.sigmoid5(self.fc5(cat_xy4+cat_xy1))  # c4

        nc_x = x.shape[1]
        nc_y = y.shape[1]
        x_a = cat_xy5[:, 0:nc_x, ...]
        x_b = cat_xy5[:, nc_x:nc_x*2, ...]
        y_a = cat_xy5[:, nc_x*2:nc_x*2+nc_y, ...]
        y_b = cat_xy5[:, nc_x*2+nc_y:nc_x*2+nc_y*2, ...]
        return x*x_a+x_b,  y*y_a+y_b


class SMAttention(nn.Module):
    def __init__(self, in_planes1, in_planes2, kernel_size=7):
        super(SMAttention, self).__init__()
        self.one_conv_x1 = nn.Conv2d(in_planes1, 1, 1)
        self.one_conv_x2 = nn.Conv2d(in_planes1, 1, 1)
        self.one_conv_y1 = nn.Conv2d(in_planes2, 1, 1)
        self.one_conv_y2 = nn.Conv2d(in_planes2, 1, 1)

        self.conv1 = nn.Conv2d(
            8, 4, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            4, 2, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            2, 2, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            2, 4, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(
            4, 4, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid5 = nn.Sigmoid()

    def forward(self, x, y):
        max_pool_x, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool_x = torch.mean(x, dim=1, keepdim=True)
        one_conv_x1 = self.one_conv_x1(x)
        one_conv_x2 = self.one_conv_x2(x)

        max_pool_y, _ = torch.max(y, dim=1, keepdim=True)
        avg_pool_y = torch.mean(y, dim=1, keepdim=True)
        one_conv_y1 = self.one_conv_y1(y)
        one_conv_y2 = self.one_conv_y2(y)

        cat_xy = torch.cat((avg_pool_x, max_pool_x,
                           one_conv_x1, one_conv_x2, avg_pool_y, max_pool_y, one_conv_y1, one_conv_y2), 1)  # c8

        cat_xy1 = self.relu1(self.conv1(cat_xy))  # c4
        cat_xy2 = self.relu2(self.conv2(cat_xy1))  # c2
        cat_xy3 = self.relu3(self.conv3(cat_xy2))  # c2
        cat_xy4 = self.relu4(self.conv4(cat_xy3))  # c4
        cat_xy5 = self.sigmoid5(self.conv5(cat_xy4+cat_xy1))  # c4

        x_a = cat_xy5[:, [0], ...]
        x_b = cat_xy5[:, [1], ...]
        y_a = cat_xy5[:, [2], ...]
        y_b = cat_xy5[:, [2], ...]

        return x*x_a+x_b,  y*y_a+y_b


class CSMABlock(nn.Module):

    def __init__(self, in_planes1, in_planes2, kernel_size=7):
        super(CSMABlock, self).__init__()
        self.cma = CMAttention(in_planes1=in_planes1, in_planes2=in_planes2)
        self.sma = SMAttention(in_planes1=in_planes1,
                               in_planes2=in_planes2, kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, y):
        x1, y1 = self.cma(x, y)
        x2, y2 = self.sma(x1, y1)
        return torch.cat((x2, y2), 1)


# --------------------------------------------
# CBMA: convolutional block attention module
# --------------------------------------------


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out+avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x*self.ca(x)
        out = out*self.sa(out)
        return out+residual


# --------------------------------------------
# ECA: Efficient Channel Attention
# --------------------------------------------


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x*y.expand_as(x)
