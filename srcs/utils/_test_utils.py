import matplotlib.pyplot as plt
import srcs.utils.utils_blurkernel_zzh as utils_blurkernel_zzh

# k = utils_deblur_zzh.linearMotionBlurKernel(
#     motion_len=30, theta=3.14*1/6, psf_sz=50)  # motion blur
k = utils_blurkernel_zzh.codedLinearMotionBlurKernel(
    motion_len=30, theta=3.14*1/6, psf_sz=50,code=[1,0,1,0,0,1,1,0,1,0])  # motion blur
plt.imshow(k, interpolation="nearest", cmap="gray")
plt.show()
