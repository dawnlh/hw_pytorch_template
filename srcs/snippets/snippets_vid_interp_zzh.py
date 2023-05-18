from tqdm import tqdm
import numpy
import PIL
import torch
from srcs.utils.utils_interp_zzh import multiCenterInterpOrder
from os.path import join as opj
import os

# ===============
# Multi-frame Central Interpolation
#   progressively generate multiple inner frames from two given frames using an interpolation algorithm which can only generate the center frame between two adjacent frames
#    for example: given frame #0 and #4, generate the frame #1, #2 and #3
# ===============


# vid_dir contains multiple sub-dirs with each sub-dir containing frames from a video
vid_dir = '/ssd/0/zzh/dataset/GoPro/GOPRO_Large_all/small_test/'
# save results to out_dir
out_dir = '/hhd/1/zzh/dataset/GoPro/GOPRO_Large_all_1920fps/small_test/'
fps_upscale = 8       # upsampling times, 8x frame rate interpolate
Model = 'paper'
# interpolation mode
model = None  # instantiate your model here and load the weights

inp_order = multiCenterInterpOrder(0, fps_upscale)

# get image dir paths
vid_names = sorted(os.listdir(vid_dir))

# loop over all videos
for vid_name in tqdm(vid_names):
    # paths
    in_vid_path = opj(vid_dir, vid_name)
    out_vid_path = opj(out_dir, vid_name)
    os.makedirs(out_vid_path)

    frame_names = sorted(os.listdir(in_vid_path))
    in_frame_paths = [opj(in_vid_path, frame_name)
                      for frame_name in frame_names]
    out_frame_paths = [opj(out_vid_path, frame_name)
                       for frame_name in frame_names]

    # processing one video
    curr_frames = [None]*(fps_upscale+1)
    for k, in_frame_path in enumerate(in_frame_paths):
        curr_frames[fps_upscale] = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(
            in_frame_path))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

        if curr_frames[0] is not None:
            # interpolate
            for m in inp_order:
                curr_frames[m[2]] = model(curr_frames[m[0]], curr_frames[m[1]])
            # save
            [out_frame_file, file_ext] = os.path.splitext(
                out_frame_paths[k-1])
            for m in range(fps_upscale):
                PIL.Image.fromarray((curr_frames[m].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[
                                    :, :, ::-1] * 255.0).astype(numpy.uint8)).save(out_frame_file+f'_{m:02d}'+file_ext)

            # load next frame
            curr_frames[0] = curr_frames[fps_upscale]

    # save the last frame
    PIL.Image.fromarray((curr_frames[0].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[
        :, :, ::-1] * 255.0).astype(numpy.uint8)).save(out_frame_paths[k])
