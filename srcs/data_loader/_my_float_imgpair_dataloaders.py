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
# loading image pairs frm data_dir and target_dir according to filename order
# dir structure: traversing all the subdirs
# output data: shape=[C,H,W], dtype=float32, range=[0,1]
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


class ImagePairDataset(Dataset):
    """
    Image dataset that loads images to CPU once per batch
    """

    def __init__(self, data_dir, target_dir, patch_size=None, tform_op=None, sigma_range=0):
        super(ImagePairDataset, self).__init__()
        self.patch_size = [patch_size] * \
            2 if isinstance(patch_size, int) else patch_size
        self.tform_op = tform_op
        self.sigma_range = sigma_range

        # get image paths
        ext = ['jpg', 'png', 'tif', 'bmp']
        self.data_paths, data_num, skip_num = get_file_path(data_dir, ext)
        self.target_paths, target_num, skip_num = get_file_path(
            target_dir, ext)
        assert data_num == target_num, f'data file num {data_num} is not equal to target file num {target_num}'
        self.dataset_num = target_num

        print(f'===> total dataset image num: {self.dataset_num}')

    def __getitem__(self, idx):
        # load data image
        datak = cv2.imread(self.data_paths[idx])
        assert datak is not None, f'Image-{self.data_paths[idx]} read falied'
        datak = cv2.cvtColor(datak, cv2.COLOR_BGR2RGB)
        datak = datak.astype(np.float32)/255
        data_sz = datak.shape
        # load target image
        targetk = cv2.imread(self.target_paths[idx])
        assert targetk is not None, 'Image-%s read falied' % self.target_paths[idx]
        targetk = cv2.cvtColor(targetk, cv2.COLOR_BGR2RGB)
        targetk = targetk.astype(np.float32)/255
        target_sz = targetk.shape
        assert data_sz == target_sz, f'data image size {data_sz} is not equal to target image size {target_sz}'

        # crop to patch size
        if self.patch_size:
            # set the random crop point
            assert (data_sz[0] >= self.patch_size[0]) and (data_sz[1] >= self.patch_size[1]
                                                           ), 'error PATCH_SZ larger than image size'
            xmin = np.random.randint(0, data_sz[1]-self.patch_size[1])
            ymin = np.random.randint(0, data_sz[0]-self.patch_size[0])

            datak = datak[ymin:ymin+self.patch_size[0],
                          xmin:xmin+self.patch_size[1], :]
            targetk = targetk[ymin:ymin+self.patch_size[0],
                              xmin:xmin+self.patch_size[1], :]

        # data augment
        if self.tform_op:
            catk = np.concatenate((datak, targetk), axis=2)
            catk = augment_img(catk, tform_op=self.tform_op)
            datak, targetk = np.split(catk, 2, axis=2)

        # add noise
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)
        assert 0 <= noise_level <= 1, f'noise level (sigma_range) should be within 0-1, but get {self.sigma_range}'
        if noise_level > 0:
            datak = datak + \
                np.random.normal(0, noise_level, datak.shape).astype(np.float32)
            datak = datak.clip(0, 1)

        # return shape=[C,H,W], dtype=float32, range=[0,1]
        return datak.transpose(2, 0, 1), targetk.transpose(2, 0, 1), noise_level

    def __len__(self):
        return self.dataset_num


class ImagePairDataset_all2CPU(Dataset):
    """
    Image dataset that loads entire dataset to CPU to speed the data load process (need larger CPU memory)
    """

    def __init__(self, data_dir, target_dir, patch_size=None, tform_op=None, sigma_range=0):
        super(ImagePairDataset_all2CPU, self).__init__()
        self.patch_size = [patch_size] * \
            2 if isinstance(patch_size, int) else patch_size
        self.tform_op = tform_op
        self.sigma_range = sigma_range
        self.data_imgs = []
        self.target_imgs = []

        # get image paths
        ext = ['jpg', 'png', 'tif', 'bmp']
        data_paths, data_num, skip_num = get_file_path(data_dir, ext)
        target_paths, target_num, skip_num = get_file_path(
            target_dir, ext)
        assert data_num == target_num, f'data file num {data_num} is not equal to target file num {target_num}'
        self.dataset_num = target_num

        # load images
        for idx in tqdm(range(target_num), desc='⏳ Loading dataset to CPU'):
            datak = cv2.imread(data_paths[idx])
            assert datak is not None, f'Image-{data_paths[idx]} read falied'
            datak = cv2.cvtColor(datak, cv2.COLOR_BGR2RGB)
            self.data_imgs.append(datak)

            targetk = cv2.imread(target_paths[idx])
            assert targetk is not None, f'Image-{target_paths[idx]} read falied'
            targetk = cv2.cvtColor(targetk, cv2.COLOR_BGR2RGB)
            self.target_imgs.append(targetk)
            assert datak.shape == targetk.shape, f'data image size {datak.shape} is not equal to target image size {targetk.shape}'

        print(f'===> total dataset image num: {self.dataset_num}')

    def __getitem__(self, idx):
        # load image
        datak = self.data_imgs[idx].astype(np.float32)/255
        targetk = self.target_imgs[idx].astype(np.float32)/255
        img_sz = datak.shape

        # crop to patch size
        if self.patch_size:
            assert (img_sz[0] >= self.patch_size[0]) and (img_sz[1] >= self.patch_size[1]
                                                          ), 'error patch_size(%d*%d) larger than image size(%d*%d)' % (*self.patch_size, *img_sz[0:2])
            xmin = np.random.randint(0, img_sz[1]-self.patch_size[1])
            ymin = np.random.randint(0, img_sz[0]-self.patch_size[0])
            datak = datak[ymin:ymin+self.patch_size[0],
                          xmin:xmin+self.patch_size[1], :]
            targetk = targetk[ymin:ymin+self.patch_size[0],
                              xmin:xmin+self.patch_size[1], :]

        # data augment
        if self.tform_op:
            catk = np.concatenate((datak, targetk), axis=2)
            catk = augment_img(catk, tform_op=self.tform_op)
            datak, targetk = np.split(catk, 2, axis=2)

        # add noise
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)
        assert 0 <= noise_level <= 1, f'noise level (sigma_range) should be within 0-1, but get {self.sigma_range}'
        if noise_level > 0:
            datak = datak + \
                np.random.normal(0, noise_level, datak.shape).astype(np.float32)
            datak = datak.clip(0, 1)

        # return shape=[C,H,W], dtype=float32, range=[0,1]
        return datak.transpose(2, 0, 1),  targetk.transpose(2, 0, 1), noise_level

    def __len__(self):
        return self.dataset_num


class ImagePairDataset_Exp:
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

def get_data_loaders(data_dir, target_dir, batch_size=8, tform_op=None, sigma_range=0, patch_size=None, shuffle=True, validation_split=0.1, status='train', num_workers=8, pin_memory=False, prefetch_factor=2, all2CPU=True):
    # dataset
    if status in ['train', 'test', 'valid']:
        if all2CPU:
            dataset = ImagePairDataset_all2CPU(
                data_dir, target_dir, patch_size, tform_op, sigma_range)
        else:
            dataset = ImagePairDataset(data_dir, target_dir,
                                       patch_size, tform_op, sigma_range)
    elif status in ['simuexp', 'realexp']:
        dataset = ImagePairDataset_Exp(
            data_dir, patch_size, tform_op, sigma_range, exp_mode=status)
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
            assert 0 <= validation_split < num_total, "validation set size is configured to be larger than entire dataset."
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
    from utils.utils_image_zzh import augment_img

    data_dir = '/ssd/0/zzh/dataset/GoPro/GOPRO_Large/test/GOPR0384_11_00/blur/'
    target_dir = '/ssd/0/zzh/dataset/GoPro/GOPRO_Large/test/GOPR0384_11_00/sharp/'
    # target_dir = data_dir

    save_dir = './outputs/tmp/test/'

    # dataloader, val_dataloader = get_data_loaders(
    #     data_dir, target_dir,  tform_op=['all'], sigma_range=0.1, patch_size=512, batch_size=1, num_workers=8, all2CPU=True)

    dataloader = get_data_loaders(data_dir, target_dir, patch_size=None, sigma_range=0.2,
                                  batch_size=1, num_workers=8, shuffle=False, all2CPU=True, status='test')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for k, in_data in enumerate(dataloader):  # val_dataloader
        datak, targetk, noise_level = in_data
        datak = datak.numpy()[0, ::-1, ...].transpose(1, 2, 0)*255
        targetk = targetk.numpy()[0, ::-1, ...].transpose(1, 2, 0)*255

        if k % 1 == 0:
            print('k = ', k)
            cv2.imwrite(opj(save_dir, '%02ddata.png' %
                            k), datak, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(opj(save_dir, '%02dtarget.png' %
                            k), targetk, [cv2.IMWRITE_PNG_COMPRESSION, 0])
