import torch
import torch.nn.functional as F
from utils.utils_patch_proc import window_partitionx, window_reversex

# ================================================
# Cope with the tensor size mismatch problem in inference
# ================================================
data = torch.randn(1, 3, 511, 513)
model = None # your model

#------------------------------------
# case1: The network expects the size of the input tensor to be an integer multiple of a specified factor (sf)
# solution: pad, inference, crop
#------------------------------------
_, _, h, w = data.shape

sf = 8  # scale_factor
# pad to a multiple of scale_factor (sf)
H, W = int((h+sf-1)/sf)*sf, int((w+sf-1)/sf)*sf
pad_h, pad_w = H-h, W-w
data_pad = F.pad(data, [0, pad_w, 0, pad_h])
output = model(data_pad) # inference
output = output[:, :, :h, :w] # crop the margin


#------------------------------------
# case2: The network expects the input tensor to be of a specific size（win_size)：
# solution: slicing window (patch) processing
#   1. using slicing window method to divide the input tensor to patchs, and stack the patches to the batch dimension
#   2. perform inference
#   3. convert the output patch stack to the tensor with origianl size
#
# https://github.com/dawnlh/my_pytorch_template/blob/main/srcs/utils/utils_patch_proc.py
#------------------------------------
_, _, h, w = data.shape
win_size = 128  # window size
data_re, batch_list = window_partitionx(data, win_size) # patch partition
output = model(data_re) # inference
output = window_reversex(output, win_size, h, w, batch_list) # patch reverse
