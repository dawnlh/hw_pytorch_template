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
# from srcs.utils.utils_image_zzh import augment_img


# =================
# loading single image
# =================

# =================
# basic functions
# =================


# =================
# Dataset
# =================


class ImageDataset(Dataset):
    """
    Image dataset that loads images to CPU once per batch
    """

    def __init__(self, data_dir, patch_size=None, tform_op=None, sigma_range=0):
        super(ImageDataset, self).__init__()
        self.patch_size = [patch_size] * \
            2 if isinstance(patch_size, int) else patch_size
        self.tform_op = tform_op
        self.sigma_range = sigma_range
        self.img_paths = []
        self.imgs = []
        self.img_num = None

        # get image paths and load images
        img_paths = []
        if isinstance(data_dir, str):
            # single dataset
            img_names = sorted(os.listdir(data_dir))
            img_paths = [opj(data_dir, img_name) for img_name in img_names]
        else:
            # multiple dataset
            for data_dir_n in sorted(data_dir):
                img_names_n = sorted(os.listdir(data_dir_n))
                img_paths_n = [opj(data_dir_n, img_name_n)
                               for img_name_n in img_names_n]
                img_paths.extend(img_paths_n)
        self.img_paths = img_paths

        # remove non-image path
        for img_path in self.img_paths:
            if img_path.split('.')[-1] not in ['jpg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % (img_path))
                self.img_paths.remove(img_path)

        self.img_num = len(self.img_paths)
        print('===> dataset image num: %d' % self.img_num)

    def __getitem__(self, idx):
        # load image
        imgk = cv2.imread(self.img_paths[idx])
        assert imgk is not None, 'Image-%s read falied' % self.img_paths[idx]
        imgk = cv2.cvtColor(imgk, cv2.COLOR_BGR2RGB)
        imgk = imgk.astype(np.float32)/255
        img_sz = imgk.shape

        # crop to patch size
        if self.patch_size:
            assert (img_sz[0] >= self.patch_size[0]) and (img_sz[1] >= self.patch_size[1]
                                                          ), 'error patch_size(%d*%d) larger than image size(%d*%d)' % (*self.patch_size, *img_sz[0:2])
            xmin = np.random.randint(0, img_sz[1]-self.patch_size[1])
            ymin = np.random.randint(0, img_sz[0]-self.patch_size[0])
            imgk = imgk[ymin:ymin+self.patch_size[0],
                        xmin:xmin+self.patch_size[1], :]
        # data augment
        if self.tform_op:
            imgk = augment_img(imgk, tform_op=self.tform_op)

        # add noise
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)
        imgk = imgk + np.random.normal(0, noise_level, imgk.shape)
        imgk = imgk.astype(np.float32).clip(0, 1)

        # [debug] test
        # multi_imsave(img*255, 'img')
        # cv2.imwrite('./outputs/tmp/test/coded_blur_img.jpg', coded_blur_img[:,:,::-1]*255)
        # cv2.imwrite('./outputs/tmp/test/clear.jpg', sharp_img[:, :, ::-1]*255)

        # return [C,H,W]
        return imgk.transpose(2, 0, 1),  noise_level

    def __len__(self):
        return self.img_num


class ImageDataset_all2CPU(Dataset):
    """
    Image dataset that loads entire dataset to CPU to speed the data load process (need larger CPU memory)
    """

    def __init__(self, data_dir, patch_size=None, tform_op=None, sigma_range=0):
        super(ImageDataset_all2CPU, self).__init__()
        self.patch_size = [patch_size] * \
            2 if isinstance(patch_size, int) else patch_size
        self.tform_op = tform_op
        self.sigma_range = sigma_range
        self.img_paths = []
        self.imgs = []
        self.img_num = None

        # get image paths and load images
        img_paths = []
        if isinstance(data_dir, str):
            # single dataset
            img_names = sorted(os.listdir(data_dir))
            img_paths = [opj(data_dir, img_name) for img_name in img_names]
        else:
            # multiple dataset
            for data_dir_n in sorted(data_dir):
                img_names_n = sorted(os.listdir(data_dir_n))
                img_paths_n = [opj(data_dir_n, img_name_n)
                               for img_name_n in img_names_n]
                img_paths.extend(img_paths_n)
        self.img_paths = img_paths

        for img_path in tqdm(self.img_paths, desc='Loading dataset to CPU'):

            if img_path.split('.')[-1] not in ['jpg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % img_path)
                continue
            img = cv2.imread(img_path)
            assert img is not None, 'Image-%s read falied' % img_path
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

        self.img_num = len(self.imgs)
        print('===> dataset image num: %d' % self.img_num)

    def __getitem__(self, idx):
        # load image
        imgk = self.imgs[idx].astype(np.float32)/255
        img_sz = imgk.shape

        # crop to patch size
        if self.patch_size:
            assert (img_sz[0] >= self.patch_size[0]) and (img_sz[1] >= self.patch_size[1]
                                                          ), 'error patch_size(%d*%d) larger than image size(%d*%d)' % (*self.patch_size, *img_sz[0:2])
            xmin = np.random.randint(0, img_sz[1]-self.patch_size[1])
            ymin = np.random.randint(0, img_sz[0]-self.patch_size[0])
            imgk = imgk[ymin:ymin+self.patch_size[0],
                        xmin:xmin+self.patch_size[1], :]
        # data augment
        if self.tform_op:
            imgk = augment_img(imgk, tform_op=self.tform_op)

        # add noise
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)
        imgk = imgk + np.random.normal(0, noise_level, imgk.shape)
        imgk = imgk.astype(np.float32).clip(0, 1)

        # [debug] test
        # multi_imsave(img*255, 'img')
        # cv2.imwrite('./outputs/tmp/test/coded_blur_img.jpg', coded_blur_img[:,:,::-1]*255)
        # cv2.imwrite('./outputs/tmp/test/clear.jpg', sharp_img[:, :, ::-1]*255)

        # return [C,H,W]
        return imgk.transpose(2, 0, 1),  noise_level

    def __len__(self):
        return self.img_num


class ImageDataset_realExp:
    """
    CE datasetfor real test (without ground truth)
    """
    pass


# =================
# get dataloader
# =================

def get_data_loaders(data_dir, batch_size=8, tform_op=None, sigma_range=0, patch_size=None, shuffle=True, validation_split=0.1, status='train', num_workers=8, pin_memory=False, prefetch_factor=2, all2CPU=True):
    # dataset
    if status == 'train' or status == 'test' or status == 'debug':
        if all2CPU:
            dataset = ImageDataset_all2CPU(
                data_dir, patch_size, tform_op, sigma_range)
        else:
            dataset = ImageDataset(data_dir, patch_size, tform_op, sigma_range)
    elif status == 'real_test':
        dataset = ImageDataset_realExp(
            data_dir, patch_size, tform_op, sigma_range)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'pin_memory': pin_memory
    }

    # dataset split & dist train assignment
    if status == 'train' or status == 'debug':
        # split dataset into train and validation set
        num_total = len(dataset)
        if isinstance(validation_split, int):
            assert validation_split > 0
            assert validation_split < num_total, "validation set size is configured to be larger than entire dataset."
            num_valid = validation_split
        else:
            num_valid = int(num_total * validation_split)
        num_train = num_total - num_valid

        train_dataset, valid_dataset = random_split(
            dataset, [num_train, num_valid])

        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            loader_args['shuffle'] = False
            train_sampler = DistributedSampler(train_dataset)
            valid_sampler = DistributedSampler(valid_dataset)
        return DataLoader(train_dataset, sampler=train_sampler, **loader_args), \
            DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
    else:
        return DataLoader(dataset, **loader_args)


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

    save_dir = './outputs/tmp/test/'

    # train_dataloader, val_dataloader = get_data_loaders(
    #     data_dir,  tform_op=['all'], sigma_range=0.1, patch_size=256, batch_size=1, num_workers=8, all2CPU=False)

    test_dataloader = get_data_loaders(
        data_dir, patch_size=None, sigma_range=0, batch_size=1, num_workers=8, shuffle=False, all2CPU=True, status='test',)

    iter_dataloader = test_dataloader

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    k = 0

    for imgk, noise_level in iter_dataloader:  # val_dataloader
        k += 1
        imgk = imgk.numpy()[0, ::-1, ...].transpose(1, 2, 0)*255

        # import matplotlib.pyplot as plt
        # plt.imshow(psf, interpolation="nearest", cmap="gray")
        # plt.show()

        if k % 1 == 0:
            print('k = ', k)
            cv2.imwrite(opj(save_dir, 'img%02d.png' %
                            k), imgk, [cv2.IMWRITE_PNG_COMPRESSION, 0])
