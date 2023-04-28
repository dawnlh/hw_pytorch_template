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
# loading blur and sharp image pairs from GoPro dataset
# =================

# =================
# basic functions
# =================

def _scan(root, blur_key='blur_gamma', sharp_key='sharp', non_blur_keys=[], non_sharp_keys=[]):
    """scan the image dataset dir for image paths
    blur_key = blur_key    # blur image's subdir name
    sharp_key = sharp_key  # sharp image's subdir name
    non_blur_keys = non_blur_keys  # excluded blur image's subdir name
    non_sharp_keys = non_sharp_keys  # excluded sharp image's subdir name
    """

    def _key_check(path, true_key, false_keys):
        path = os.path.join(path, '')
        if path.find(true_key) >= 0:
            for false_key in false_keys:
                if path.find(false_key) >= 0:
                    return False
            return True
        else:
            return False

    def _get_list_by_key(root, true_key, false_keys):
        data_list = []
        for sub, dirs, files in os.walk(root):
            if not dirs:
                file_list = [os.path.join(sub, f) for f in files]
                if _key_check(sub, true_key, false_keys):
                    data_list += file_list

        data_list.sort()
        return data_list

    # rectify keys
    if blur_key in non_blur_keys:
        non_blur_keys.remove(blur_key)
    if sharp_key in non_sharp_keys:
        non_sharp_keys.remove(sharp_key)

    blur_key = os.path.join(blur_key, '')
    non_blur_keys = [os.path.join(
        non_blur_key, '') for non_blur_key in non_blur_keys]
    sharp_key = os.path.join(sharp_key, '')
    non_sharp_keys = [os.path.join(
        non_sharp_key, '') for non_sharp_key in non_sharp_keys]

    # get path list
    blur_list = _get_list_by_key(
        root, blur_key, non_blur_keys)
    sharp_list = _get_list_by_key(
        root, sharp_key, non_sharp_keys)

    return blur_list, sharp_list

# =================
# Dataset
# =================


class GoproDataset(Dataset):
    """
    GoPro dataset that loads images to CPU once per batch
    """

    def __init__(self, data_dir, patch_size=None, tform_op=None, sigma_range=0, blur_key='blur_gamma', sharp_key='sharp', non_blur_keys=[], non_sharp_keys=[]):
        super(GoproDataset, self).__init__()
        self.patch_size = [patch_size] * \
            2 if isinstance(patch_size, int) else patch_size
        self.tform_op = tform_op
        self.sigma_range = sigma_range
        self.img_num = None

        # get image paths
        self.blur_list, self.sharp_list = _scan(root=data_dir, blur_key=blur_key, sharp_key=sharp_key,
                                                non_blur_keys=non_blur_keys, non_sharp_keys=non_sharp_keys)

        # dataset info
        assert(len(self.blur_list) == len(self.sharp_list)
               ), f'The number of blur images ({len(self.blur_list)}) is not equal to that of sharp images ({len(self.sharp_list)})'
        self.img_num = len(self.blur_list)
        print('===> dataset image num: %d' % self.img_num)

    def __getitem__(self, idx):
        # load image
        imgk = cv2.imread(self.blur_list[idx])
        assert imgk is not None, 'Image-%s read falied' % self.blur_list[idx]
        sharpk_ = cv2.imread(self.sharp_list[idx])
        assert sharpk_ is not None, 'Image-%s read falied' % self.sharp_list[idx]
        # cat the blur and sharp image
        imgk = np.concatenate([imgk, sharpk_], axis=2)

        # crop to patch size
        img_sz = sharpk_.shape
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

        # cvtcolor & dtype
        blurk = imgk[:, :, 0:3]
        blurk = cv2.cvtColor(blurk, cv2.COLOR_BGR2RGB)
        blurk = blurk.astype(np.float32)/255
        sharpk = imgk[:, :, 3:]
        sharpk = cv2.cvtColor(sharpk, cv2.COLOR_BGR2RGB)
        sharpk = sharpk.astype(np.float32)/255

        # add noise
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)
        blurk = blurk + \
            np.random.normal(0, noise_level, blurk.shape).astype(np.float32)
        blurk = blurk.clip(0, 1)

        # [debug] test
        # cv2.imwrite('./outputs/tmp/test/blur.jpg', blurk[:, :, ::-1]*255)
        # cv2.imwrite('./outputs/tmp/test/sharp.jpg', sharpk[:, :, ::-1]*255)

        # return [C,H,W]
        return blurk.transpose(2, 0, 1),  sharpk.transpose(2, 0, 1), noise_level

    def __len__(self):
        return self.img_num


class GoproDataset_all2CPU(Dataset):
    """
    GoPro dataset that loads entire dataset to CPU to speed the data load process (need larger CPU memory)
    """

    def __init__(self, data_dir, patch_size=None, tform_op=None, sigma_range=0, blur_key='blur_gamma', sharp_key='sharp', non_blur_keys=[], non_sharp_keys=[]):
        super(GoproDataset_all2CPU, self).__init__()
        self.patch_size = [patch_size] * \
            2 if isinstance(patch_size, int) else patch_size
        self.tform_op = tform_op
        self.sigma_range = sigma_range
        self.img_num = None

        # get image paths
        self.blur_list, self.sharp_list = _scan(root=data_dir, blur_key=blur_key, sharp_key=sharp_key,
                                                non_blur_keys=non_blur_keys, non_sharp_keys=non_sharp_keys)
        assert(len(self.blur_list) == len(self.sharp_list)
               ), 'The number of blur images is not equal to that of sharp images'

        # load images
        self.blur = []
        self.sharp = []
        for img_path in tqdm(self.blur_list, desc='Loading blur images to CPU'):
            if img_path.split('.')[-1] not in ['jpg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % img_path)
                continue
            img = cv2.imread(img_path)
            assert img is not None, 'Image-%s read falied' % img_path
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.blur.append(img)

        for img_path in tqdm(self.sharp_list, desc='Loading sharp images to CPU'):
            if img_path.split('.')[-1] not in ['jpg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % img_path)
                continue
            img = cv2.imread(img_path)
            assert img is not None, 'Image-%s read falied' % img_path
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.sharp.append(img)

        # dataset info
        self.img_num = len(self.blur_list)
        print('===> dataset image num: %d' % self.img_num)

    def __getitem__(self, idx):
        # load image
        imgk = np.concatenate([self.blur[idx], self.sharp[idx]], axis=2)
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

        # split
        blurk = imgk[:, :, 0:3]
        sharpk = imgk[:, :, 3:]

        # add noise
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)
        blurk = blurk + \
            np.random.normal(0, noise_level, blurk.shape).astype(np.float32)
        blurk = blurk.clip(0, 1)

        # [debug] test
        # cv2.imwrite('./outputs/tmp/test/coded_blur_img.jpg', coded_blur_img[:,:,::-1]*255)
        # cv2.imwrite('./outputs/tmp/test/clear.jpg', sharp_img[:, :, ::-1]*255)

        # return [C,H,W]
        return blurk.transpose(2, 0, 1),  sharpk.transpose(2, 0, 1), noise_level

    def __len__(self):
        return self.img_num


class GoproDataset_realExp:
    """
    CE datasetfor real test (without ground truth)
    """
    pass


# =================
# get dataloader
# =================

def get_data_loaders(data_dir, batch_size=8, tform_op=None, sigma_range=0, blur_key='blur_gamma', patch_size=None, shuffle=True, validation_split=0.1, status='train', num_workers=8, pin_memory=False, prefetch_factor=2, all2CPU=True):
    # dataset
    if status == 'train' or status == 'test' or status == 'debug':
        if all2CPU:
            dataset = GoproDataset_all2CPU(
                data_dir, patch_size, tform_op, sigma_range, blur_key=blur_key)
        else:
            dataset = GoproDataset(data_dir, patch_size,
                                   tform_op, sigma_range, blur_key=blur_key)
    elif status == 'realexp':
        dataset = GoproDataset_realExp(
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

    data_dir = '/ssd/0/zzh/dataset/GoPro/test/'

    save_dir = './outputs/tmp/test/'

    train_dataloader, val_dataloader = get_data_loaders(
        data_dir,  tform_op=['all'], sigma_range=0.1, patch_size=256, batch_size=1, num_workers=8, all2CPU=True)

    # test_dataloader = get_data_loaders(
    #     data_dir, patch_size=None, sigma_range=0.02, batch_size=1, num_workers=8, shuffle=False, all2CPU=True, status='test')

    iter_dataloader = val_dataloader

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    k = 0

    for imgk, gtk, noise_level in iter_dataloader:  # val_dataloader
        k += 1
        imgk = imgk.numpy()[0, ::-1, ...].transpose(1, 2, 0)*255
        gtk = gtk.numpy()[0, ::-1, ...].transpose(1, 2, 0)*255

        # import matplotlib.pyplot as plt
        # plt.imshow(psf, interpolation="nearest", cmap="gray")
        # plt.show()

        if k % 5 == 0:
            print('k = ', k)
            cv2.imwrite(opj(save_dir, '%02dimg.png' %
                            k), imgk, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(opj(save_dir, '%02dgt.png' %
                            k), gtk, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if k == 250:
            break
