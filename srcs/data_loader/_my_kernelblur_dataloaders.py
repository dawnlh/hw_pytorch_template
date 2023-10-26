import sys
import torch.distributed as dist
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from scipy import ndimage
import cv2
import os
import numpy as np
from tqdm import tqdm
from os.path import join as opj
from srcs.utils import utils_blurkernel_zzh
import torch.nn.functional as F
# from srcs.utils.utils_image_zzh import augment_img


# =================
# loading single image from a directory or multiple directories, and blurring it with kernels
#
# data dir structure:
#     data_dir1
#     ├─ img1
#     ├─ img2
#     ├─ ...
# =================

# =================
# basic functions
# =================


def init_network_input(coded_blur_img, code):
    """
    calculate the initial input of the network

    Args:
        coded_blur_img (ndarray): coded measurement
        code (ndarray): encoding code
    """
    return coded_blur_img


def img_blur(img, psf, noise_level=0.01, mode='circular', cval=0):
    """
    blur image with blur kernel

    Args:
        img (ndarray): gray or rgb sharp image,[H, W <,C>]  (0-1)
        psf (ndarray): blur kernel,[H, W <,C>]
        noise_level (scalar): gaussian noise std (0-1)
        mode (str): convolution mode, 'circular' ('wrap') | 'constant' | ...
        cval: padding value for 'constant' padding

        refer to ndimage.filters.convolve for specific param setting

    Returns:
        x: blurred image
    """
    # convolution
    if mode == 'circular':
        # in ndimage.filters.convolve, 'circular'=='wrap'
        mode = 'wrap'
    if img.ndim == 3 and psf.ndim == 2:
        # rgb image
        psf = np.expand_dims(psf, axis=2)

    blur_img = ndimage.filters.convolve(
        img, psf, mode=mode, cval=cval)

    # add Gaussian noise
    blur_noisy_img = blur_img + \
        np.random.normal(0, noise_level, blur_img.shape)
    return blur_noisy_img.astype(np.float32).clip(0, 1)


# =================
# Dataset
# =================


class BlurImgDataset(Dataset):
    """
    generate blur image from shape image, load sample during forward
    """

    def __init__(self, data_dir, ce_code=None, patch_sz=None, tform_op=None, sigma_range=0, motion_len=0, load_psf_dir=None):
        super(BlurImgDataset, self).__init__()
        self.ce_code = ce_code
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.sigma_range = sigma_range
        self.motion_len = motion_len
        # use loaded psf, rather than generated
        self.load_psf = True if load_psf_dir else False
        self.img_paths = []
        self.img_num = None

        # get image paths and load images
        img_paths = []
        if isinstance(data_dir, str):
            # single dataset
            img_names = sorted(os.listdir(data_dir))
            img_paths = [opj(data_dir, vid_name) for vid_name in img_names]
        else:
            # multiple dataset
            for data_dir_n in sorted(data_dir):
                img_names_n = sorted(os.listdir(data_dir_n))
                img_paths_n = [opj(data_dir_n, img_name_n)
                               for img_name_n in img_names_n]
                img_paths.extend(img_paths_n)
        self.img_paths = img_paths

        for img_path in self.img_paths:
            if img_path.split('.')[-1] not in ['jpg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % (img_path))
                self.img_paths.remove(img_path)
        self.img_num = len(self.img_paths)
        print('===> dataset image num: %d' % self.img_num)

        # get loaded psf paths and load psfs
        if self.load_psf:
            psf_names = sorted(os.listdir(load_psf_dir))
            self.psf_paths = [opj(load_psf_dir, psf_name)
                              for psf_name in psf_names]
            for psf_path in self.psf_paths:
                if psf_path.split('.')[-1] not in ['jpg', 'png', 'tif', 'bmp']:
                    print('Skip a non-image file:%s' % (psf_path))
                    self.psf_paths.remove(psf_path)
            self.psf_num = len(psf_names)
            print('===> dataset psf num: %d' % self.psf_num)

    def __getitem__(self, idx):
        # load image and traj
        imgk = cv2.imread(self.img_paths[idx])
        assert imgk is not None, 'Image-%s read falied' % self.img_paths[idx]
        imgk = cv2.cvtColor(imgk, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        img_sz = imgk.shape

        # crop to patch size
        if self.patch_sz:
            assert (img_sz[0] >= self.patch_sz[0]) and (img_sz[1] >= self.patch_sz[1]
                                                        ), 'error patch_size(%d*%d) larger than image size(%d*%d)' % (*self.patch_sz, *img_sz[0:2])
            xmin = np.random.randint(0, img_sz[1]-self.patch_sz[1])
            ymin = np.random.randint(0, img_sz[0]-self.patch_sz[0])
            imgk = imgk[ymin:ymin+self.patch_sz[0],
                        xmin:xmin+self.patch_sz[1], :]
        # data augment
        if self.tform_op:
            imgk = augment_img(imgk, tform_op=self.tform_op)

        # get/load psf
        if self.load_psf:
            if self.psf_num < self.img_num:
                psf_idx = idx*self.psf_num//self.img_num
            else:
                psf_idx = idx
            psfk = cv2.imread(self.psf_paths[psf_idx])
            assert psfk is not None, 'PSF-%s read falied' % self.psf_paths[idx]
            psfk = cv2.cvtColor(psfk, cv2.COLOR_BGR2GRAY)
            psfk = psfk.astype(np.float32)/np.sum(psfk)
        else:
            # generate random psf
            psfk, _ = utils_blurkernel_zzh.codedRandomMotionBlurKernel(
                motion_len_r=self.motion_len, psf_sz=64, code=self.ce_code)
            # psf = utils_deblur_kair.blurkernel_synthesis()

        # energy normalize
        if self.ce_code:
            psfk = psfk*sum(self.ce_code)/len(self.ce_code)

        # noise level
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)

        # blur and add noise
        coded_blur_img = img_blur(imgk, psfk, noise_level, mode='constant')

        # psf expand divmod(divident, divisor)
        psfk = np.expand_dims(np.float32(psfk), axis=2)
        noise_level = np.reshape(noise_level, [1, 1, 1]).astype(np.float32)
        # [debug] test
        # multi_imsave(vid*255, 'vid')
        # cv2.imwrite('./outputs/tmp/test/coded_blur_img.jpg', coded_blur_img[:,:,::-1]*255)
        # cv2.imwrite('./outputs/tmp/test/clear.jpg', sharp_img[:, :, ::-1]*255)

        # return [C,H,W]
        return coded_blur_img_noisy.transpose(2, 0, 1), psfk.transpose(2, 0, 1), imgk.transpose(2, 0, 1), coded_blur_img.transpose(2, 0, 1),  noise_level

    def __len__(self):
        return self.img_num


class BlurImgDataset_all2CPU(Dataset):
    """
    generate blur image from shape image, load entire dataset to CPU to speed up the data load process
    """

    def __init__(self, data_dir, ce_code=None, patch_sz=None, tform_op=None, sigma_range=0, motion_len=0, load_psf_dir=None):
        super(BlurImgDataset_all2CPU, self).__init__()
        self.ce_code = ce_code
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.sigma_range = sigma_range
        self.motion_len = motion_len
        # use loaded psf, rather than generated
        self.load_psf = True if load_psf_dir else False
        self.img_paths = []
        self.imgs = []
        self.psfs = []
        self.img_num = None

        # get image paths and load images
        img_paths = []
        if isinstance(data_dir, str):
            # single dataset
            img_names = sorted(os.listdir(data_dir))
            img_paths = [opj(data_dir, vid_name) for vid_name in img_names]
        else:
            # multiple dataset
            for data_dir_n in sorted(data_dir):
                img_names_n = sorted(os.listdir(data_dir_n))
                img_paths_n = [opj(data_dir_n, vid_name_n)
                               for vid_name_n in img_names_n]
                img_paths.extend(img_paths_n)
        self.img_paths = img_paths

        for img_path in tqdm(self.img_paths, desc='⏳ Loading image to CPU'):

            if img_path.split('.')[-1] not in ['jpg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file')
                continue
            img = cv2.imread(img_path)
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

        self.img_num = len(self.imgs)

        # get loaded psf paths and load psfs
        if self.load_psf:
            psf_names = sorted(os.listdir(load_psf_dir))
            self.psf_num = len(psf_names)

            for psf_name in tqdm(psf_names, desc='⏳ Loading psf to CPU'):
                psf_path = opj(load_psf_dir, psf_name)
                if psf_path.split('.')[-1] not in ['jpg', 'png', 'tif', 'bmp']:
                    print('Skip a non-image file')
                    continue
                psf = cv2.imread(psf_path)
                assert psf is not None, 'Image read falied'
                psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
                psf = psf.astype(np.float32)/np.sum(psf)
                self.psfs.append(psf)

    def __getitem__(self, idx):
        # load sharp image
        imgk = np.array(self.imgs[idx], dtype=np.float32)/255
        img_sz = imgk.shape
        # crop to patch size
        if self.patch_sz:
            assert (img_sz[0] >= self.patch_sz[0]) and (img_sz[1] >= self.patch_sz[1]
                                                        ), 'error PATCH_SZ(%d*%d) larger than image size(%d*%d)' % (*self.patch_sz, *img_sz[0:2])
            xmin = np.random.randint(0, img_sz[1]-self.patch_sz[1])
            ymin = np.random.randint(0, img_sz[0]-self.patch_sz[0])
            imgk = imgk[ymin:ymin+self.patch_sz[0],
                        xmin:xmin+self.patch_sz[1], :]
        # data augment
        if self.tform_op:
            imgk = augment_img(imgk, tform_op=self.tform_op)

        # get psf, noise level and calc blur image
        if self.load_psf:
            if self.psf_num < self.img_num:
                psf_idx = idx*self.psf_num//self.img_num
            else:
                psf_idx = idx
            # load psf from image
            psfk = self.psfs[psf_idx]
            # psf = psf[1:,1:] # for odd size psf
        else:
            # generate random psf
            # # psf = utils_deblur_zzh.codedLinearMotionBlurKernel(
            #     motion_len=self.motion_len, psf_sz=64, code=self.ce_code)  # motion blur
            psfk, _ = utils_blurkernel_zzh.codedRandomMotionBlurKernel(
                motion_len_r=self.motion_len, psf_sz=64, code=self.ce_code)
            # psf = utils_deblur_kair.blurkernel_synthesis()

        # energy normalize
        if self.ce_code:
            psfk = psfk*sum(self.ce_code)/len(self.ce_code)

        # noise level
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)

        # blur and add noise
        coded_blur_img = img_blur(imgk, psfk, noise_level, mode='constant')

        # data arange
        psfk = np.expand_dims(np.float32(psfk), axis=2)
        noise_level = np.reshape(noise_level, [1, 1, 1]).astype(np.float32)
        coded_blur_img_noisy = coded_blur_img_noisy.astype(np.float32)
        # [debug] test
        # multi_imsave(vid*255, 'vid')
        # cv2.imwrite('./outputs/tmp/test/coded_blur_img.jpg', coded_blur_img[:,:,::-1]*255)
        # cv2.imwrite('./outputs/tmp/test/clear.jpg', sharp_img[:, :, ::-1]*255)

        # return [C,H,W]
        return coded_blur_img_noisy.transpose(2, 0, 1), psfk.transpose(2, 0, 1), imgk.transpose(2, 0, 1), coded_blur_img.transpose(2, 0, 1),  noise_level

    def __len__(self):
        return self.img_num


class BlurImgDataset_RealExp:
    """
    Dataset for experiment
    expmode:
        simuexp: with ground truth
        realexp: without ground truth
    """
    pass


# =================
# get dataloader
# =================

def get_data_loaders(data_dir, ce_code=None, patch_size=None, batch_size=8, tform_op=None, sigma_range=0.05, motion_len=[10, 20], load_psf_dir=None, shuffle=True, validation_split=0.1, status='train', num_workers=8, pin_memory=False, prefetch_factor=2, all2CPU=True):
    # dataset
    if status in ['train', 'test', 'valid']:
        if all2CPU:
            dataset = BlurImgDataset_all2CPU(
                data_dir, ce_code, patch_size, tform_op, sigma_range, motion_len, load_psf_dir=load_psf_dir)
        else:
            dataset = BlurImgDataset(
                data_dir, ce_code, patch_size, tform_op, sigma_range, motion_len, load_psf_dir=load_psf_dir)
    elif status in ['simuexp', 'realexp']:
        dataset = BlurImgDataset_RealExp(data_dir, ce_code, patch_size)
    else:
        raise NotImplementedError(
            f"status ({status}) should be 'train' | 'test' | 'simuexp' | 'realexp'")

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'pin_memory': pin_memory
    }

    # dataset split & dist train assignment
    if status == 'train':
        # split dataset into train and validation set
        num_total = len(dataset)
        if isinstance(validation_split, int):
            assert validation_split >= 0
            assert validation_split < num_total, "validation set size is configured to be larger than entire dataset."
            num_valid = validation_split
        elif isinstance(validation_split, float):
            num_valid = int(num_total * validation_split)
        else:
            num_valid = 0  # don't split valid set

        num_train = num_total - num_valid

        train_dataset, valid_dataset = random_split(
            dataset, [num_train, num_valid])

        # distribution trainning setting
        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            loader_args['shuffle'] = False
            train_sampler = DistributedSampler(train_dataset)
            if num_valid != 0:
                valid_sampler = DistributedSampler(valid_dataset)

        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, **loader_args)
        if num_valid != 0:
            val_dataloader = DataLoader(
                valid_dataset, sampler=valid_sampler, **loader_args)
        else:
            val_dataloader = []

        return train_dataloader, val_dataloader

    elif status in ['test', 'realexp', 'simuexp']:
        return DataLoader(dataset, **loader_args)

    elif status == 'valid':
        if dist.is_initialized():
            loader_args['shuffle'] = False
            sampler = DistributedSampler(dataset)
        return DataLoader(dataset, sampler=sampler, **loader_args)
    else:
        raise(ValueError(
            "$Status can only be 'train'|'test'|'valid'|'simuexp'|'realexp'"))

if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__) + os.sep + '../')
    from utils import utils_deblur_zzh
    from utils import utils_deblur_kair
    from utils.utils_image_zzh import augment_img

    # data_dir = '/ssd/2/zzh/dataset/Flickr2K_HR/'
    # data_dir = '/ssd/2/zzh/dataset/BSDS500_images/val/'
    data_dir = '/ssd/2/zzh/dataset/CBSD68/'
    # data_dir = '/ssd/2/zzh/dataset/DIV2K_valid_HR/'
    # data_dir = '/ssd/2/zzh/dataset/Waterloo_Exploration_Database/images/'

    # load_psf_dir = '/hdd/1/zzh/project/CED-Net/dataset/benchmark/pair_traj_psf1/box_psf/'
    load_psf_dir = None

    save_dir = './outputs/tmp/test/'

    # train_dataloader, val_dataloader = get_data_loaders(
    #     data_dir, ce_code=None, patch_size=256, tform_op=['all'], sigma_range=0, motion_len=[25, 48], batch_size=1, num_workers=8, all2CPU=False)

    test_dataloader = get_data_loaders(
        data_dir, patch_size=None, load_psf_dir=load_psf_dir, sigma_range=0.03,  motion_len=[25, 48], batch_size=1, num_workers=8, shuffle=False, all2CPU=False, status='test')

    iter_dataloader = test_dataloader

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    k = 0

    for coded_blur_img_noisy, psf, sharp_img, coded_blur_img, noise_level in iter_dataloader:  # val_dataloader
        k += 1
        coded_blur_img_noisy = coded_blur_img_noisy.numpy(
        )[0, ::-1, ...].transpose(1, 2, 0)*255
        coded_blur_img = coded_blur_img.numpy(
        )[0, ::-1, ...].transpose(1, 2, 0)*255
        sharp_img = sharp_img.numpy()[0, ::-1, ...].transpose(1, 2, 0)*255
        psf = psf.numpy()[0].transpose(1, 2, 0)
        psf = psf/np.max(psf)*255

        # import matplotlib.pyplot as plt
        # plt.imshow(psf, interpolation="nearest", cmap="gray")
        # plt.show()

        if k % 1 == 0:
            print('k = ', k)
            cv2.imwrite(opj(save_dir, 'coded_blur_img_noisy%02d.png' %
                            k), coded_blur_img_noisy, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(opj(save_dir, 'coded_blur_img%02d.png' %
                            k), coded_blur_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(opj(save_dir, 'clear%02d.png' %
                            k), sharp_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(opj(save_dir, 'psf%02d.png' %
                            k), psf, [cv2.IMWRITE_PNG_COMPRESSION, 0])
