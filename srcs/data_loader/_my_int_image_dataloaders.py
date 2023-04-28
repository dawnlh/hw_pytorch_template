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
from srcs.utils.utils_image_zzh import augment_img


# =================
# loading single image
# dir structure: traversing all the subdirs
# output data: shape=[C,H,W], dtype=uint8/uint16
# =================

# =================
# basic functions
# =================
def file_traverse(dir, ext=None):
    """
    traverse all the files and get their paths
    Args:
        dir (str): root dir path
        ext (list[str], optional): included file extensions. Defaults to None, meaning inculding all files.
    """

    data_paths = []
    skip_num = 0
    file_num = 0

    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            img_path = opj(dirpath, filename)
            if ext and img_path.split('.')[-1] not in ext:
                print('Skip a file: %s' % (img_path))
                skip_num += 1
            else:
                data_paths.append(img_path)
                file_num += 1
    return sorted(data_paths), file_num, skip_num


def get_file_path(data_dir, ext=None):
    """
    Get file paths for given directory or directories

    Args:
        data_dir (str): root dir path
        ext (list[str], optional): included file extensions. Defaults to None, meaning inculding all files.
    """

    if isinstance(data_dir, str):
        # single dataset
        data_paths, file_num, skip_num = file_traverse(data_dir, ext)
    elif isinstance(data_dir, list):
        # multiple datasets
        data_paths, file_num, skip_num = [], 0, 0
        for data_dir_n in sorted(data_dir):
            data_paths_n, file_num_n, skip_num_n = file_traverse(
                data_dir_n, ext)
            data_paths.extend(data_paths_n)
            file_num += file_num_n
            skip_num += skip_num_n
    else:
        raise ValueError('data dir should be a str or a list of str')

    return sorted(data_paths), file_num, skip_num

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

        # get image paths and load images
        ext = ['jpg', 'png', 'tif', 'bmp']
        self.img_paths, self.img_num, skip_num = get_file_path(data_dir, ext)
        print(f'===> total dataset image num: {self.img_num}')

    def __getitem__(self, idx):
        # load image
        imgk = cv2.imread(self.img_paths[idx])
        assert imgk is not None, 'Image-%s read falied' % self.img_paths[idx]
        imgk = cv2.cvtColor(imgk, cv2.COLOR_BGR2RGB)
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
        assert 0 <= noise_level <= 1, f'noise level (sigma_range) should be within 0-1, but get {self.sigma_range}'
        if noise_level > 0:
            image_dtype = imgk.dtype
            # maxv: 255/65535 for uint8/uint16
            image_maxv = np.iinfo(image_dtype).max
            imgk = imgk + np.random.normal(0, image_maxv*noise_level, imgk.shape)
            imgk = imgk.clip(0, image_maxv).astype(image_dtype)

        # imgk: [C,H,W], uint8/uint16, noise_level:0-1
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

        # get image paths
        ext = ['jpg', 'png', 'tif', 'bmp']
        img_paths, self.img_num, skip_num = get_file_path(data_dir, ext)

        # load images
        for img_path in tqdm(img_paths, desc='Loading dataset to CPU'):
            img = cv2.imread(img_path)
            assert img is not None, 'Image-%s read falied' % img_path
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

        self.img_num = len(self.imgs)
        print('===> dataset image num: %d' % self.img_num)

    def __getitem__(self, idx):
        # load image
        imgk = self.imgs[idx]
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
        assert 0 <= noise_level <= 1, f'noise level (sigma_range) should be within 0-1, but get {self.sigma_range}'
        if noise_level > 0:
            image_dtype = imgk.dtype
            # maxv: 255/65535 for uint8/uint16
            image_maxv = np.iinfo(image_dtype).max
            imgk = imgk + \
                np.random.normal(0, image_maxv*noise_level, imgk.shape)
            imgk = imgk.clip(0, image_maxv).astype(image_dtype)

        # imgk: [C,H,W], uint8/uint16, noise_level:0-1
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
    from utils.utils_image_zzh import augment_img

    data_dir = '/ssd/0/zzh/dataset/GoPro/GOPRO_Large/test/GOPR0384_11_00/sharp/'

    save_dir = './outputs/tmp/test/'

    dataloader, val_dataloader = get_data_loaders(
        data_dir,  tform_op=['all'], sigma_range=0.1, patch_size=256, batch_size=1, num_workers=8, all2CPU=False)

    # dataloader = get_data_loaders(
    #     data_dir, patch_size=None, sigma_range=0, batch_size=1, num_workers=8, shuffle=False, all2CPU=True, status='test')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for k, in_data in enumerate(dataloader):  # val_dataloader
        imgk, noise_level = in_data

        imgk = imgk.numpy()[0, ::-1, ...].transpose(1, 2, 0)*255

        if k % 1 == 0:
            print('k = ', k)
            cv2.imwrite(opj(save_dir, 'img%02d.png' %
                            k), imgk, [cv2.IMWRITE_PNG_COMPRESSION, 0])
