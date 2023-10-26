import sys
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
import cv2
import os
import numpy as np
from tqdm import tqdm
from os.path import join as opj
from srcs.utils.utils_image_zzh import augment_img
from srcs.utils.utils_noise_zzh import BasicNoiseModel, CMOS_Camera

# =================
# loading single images and add noise
# =================

# =================
# basic functions
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

def init_network_input(img):
    """
    calculate the initial input of the network

    Args:
        img (ndarray): img
    """
    return img


def img_saturation(img, mag_times=1.2, min=0, max=1):
    """
    saturation generation by magnify and clip
    """
    # return np.clip(img*mag_times, min, max)
    return np.clip(img*mag_times, min, max)/mag_times

# =================
# Dataset
# =================


class ImgNoiseDataset(Dataset):
    """
    generate noisy images from loaded sharp images, load samples during each iteration
    """

    def __init__(self, img_dir, patch_sz=256, tform_op=None, noise_type='gaussian', noise_params={'sigma': 0}):
        super(ImgNoiseDataset, self).__init__()
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.noise_type = noise_type
        self.img_paths = []
        self.img_num = None
        self.noise_params = noise_params
        if noise_type == 'gaussian':
            self.noise_model = BasicNoiseModel(noise_type, noise_params)
        elif noise_type == 'camera':
            self.noise_model = CMOS_Camera(noise_params)

        # get image paths and load images
        ext = ['jpg', 'png', 'tif', 'bmp']
        self.img_paths, self.img_num, skip_num = get_file_path(img_dir, ext)
        print(f'===> total dataset image num: {self.img_num}')


    def __getitem__(self, idx):
        # index for load image and psf
        img_idx = idx

        # load image
        imgk = cv2.imread(self.img_paths[img_idx])
        assert imgk is not None, 'Image-%s read falied' % self.img_paths[img_idx]
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

        # add noise
        if self.noise_type == 'gaussian':
            img_noisy, _ = self.noise_model.add_noise(
                imgk)
        elif self.noise_type == 'camera':
            self.noise_model.uniform_sampling_noise_params(
                self.noise_params)  # uniform sampling parameters
            kc = self.noise_params['kc']
            kc = kc if isinstance(kc, (int, float)) else np.floor(np.random.uniform(
                *kc))
            img_noisy = self.noise_model.take_photo_P(
                imgk*255, imgsize=imgk.shape, kd=kc/6, ka=6)/255

        # return [C,H,W]
        return img_noisy.transpose(2, 0, 1).astype(np.float32), imgk.transpose(2, 0, 1), self.noise_params

    def __len__(self):
        return self.img_num


class ImgNoiseDataset_all2CPU(Dataset):
    """
    generate noisy images from loaded sharp images, load entire dataset to CPU to speed the data load process
    """

    def __init__(self, img_dir, patch_sz=256, tform_op=None, noise_type='gaussian', noise_params={'sigma': 0}):
        super(ImgNoiseDataset_all2CPU, self).__init__()
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.noise_type = noise_type
        self.noise_params = noise_params
        if noise_type == 'gaussian':
            self.noise_model = BasicNoiseModel(noise_type, noise_params)
        elif noise_type == 'camera':
            self.noise_model = CMOS_Camera(noise_params)
        self.img_paths = []
        self.imgs = []
        self.psfs = []
        self.img_num = None

        # get image paths
        ext = ['jpg', 'png', 'tif', 'bmp']
        self.img_paths, self.img_num, skip_num = get_file_path(img_dir, ext)
        # load images
        for img_path in tqdm(self.img_paths, desc='⏳ Loading image to CPU'):
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
        if self.noise_type == 'gaussian':
            img_noisy, _ = self.noise_model.add_noise(imgk)
        elif self.noise_type == 'camera':
            self.noise_model.uniform_sampling_noise_params(
                self.noise_params)  # uniform sampling parameters
            kc = self.noise_params['kc']
            kc = kc if isinstance(kc, (int, float)) else np.floor(np.random.uniform(
                *kc))
            img_noisy = self.noise_model.take_photo_P(
                imgk*255, imgsize=imgk.shape, kd=kc/6, ka=6)/255


        # return [C,H,W]
        return img_noisy.transpose(2, 0, 1).astype(np.float32), imgk.transpose(2, 0, 1), self.noise_params

    def __len__(self):
        return self.img_num


class ImgNoiseDataset_Exp_all2CPU(Dataset):
    """
    load image and ground truth (for 'simuexp' exp) for normal experiments, load entire dataset to CPU to speed the data load process. (image format data)
    exp_mode:
        - simuexp: with gt
        - realexp: no gt   
    patch_sz: assign image size of patch processing to save GPU memory, default = None, use whole image
    """

    def __init__(self, noisy_img_dir, gt_img_dir=None, patch_sz=None, exp_mode='simuexp'):
        super(ImgNoiseDataset_Exp_all2CPU, self).__init__()
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        # use loaded psf, rather than generated
        self.img_dir, self.gt_img_dir = noisy_img_dir, gt_img_dir
        self.exp_mode = exp_mode
        self.imgs = []
        self.gts = []

        # get image paths and load images
        img_paths = []
        img_names = sorted(os.listdir(noisy_img_dir))
        img_paths = [opj(noisy_img_dir, img_name) for img_name in img_names]
        self.img_num = len(img_paths)

        for img_path in tqdm(img_paths, desc='⏳ Loading image to CPU'):

            if img_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % (img_path))
                continue
            img = cv2.imread(img_path)
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

        # get gt paths and load gts
        if self.exp_mode == 'simuexp':
            gt_paths = []
            gt_names = sorted(os.listdir(gt_img_dir))
            gt_paths = [opj(gt_img_dir, gt_name) for gt_name in gt_names]
            self.gt_num = len(gt_paths)

            for gt_path in tqdm(gt_paths, desc='⏳ Loading gt to CPU'):
                if gt_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                    print('Skip a non-image file: %s' % (gt_path))
                    continue
                gt = cv2.imread(gt_path)
                assert gt is not None, 'Image read falied'
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                self.gts.append(gt)

    def real_data_preproc(self, img):
        # Reducing boundary artifacts in image deconvolution
        # img_wrap = edgetaper_np(img, kernel)

        H, W = img.shape[0:2]
        H1, W1 = np.int32(H/2), np.int32(W/2)
        img_wrap = np.pad(
            img/1.2, ((H1, H1), (W1, W1), (0, 0)), mode='symmetric')

        return img_wrap

    def __getitem__(self, idx):

        # load noisy data
        _imgk = np.array(self.imgs[idx], dtype=np.float32)/255
        if self.exp_mode == 'simuexp':
            imgk = _imgk
            gtk = np.array(self.gts[idx], dtype=np.float32)/255

        elif self.exp_mode == 'realexp':
            # imgk = _imgk
            # imgk = np.expand_dims(_imgk, 2).repeat(3, 2)
            imgk = self.real_data_preproc(_imgk)
            gtk = np.zeros_like(imgk, dtype=np.float32)

        return imgk.transpose(2, 0, 1).astype(np.float32), gtk.transpose(2, 0, 1), []

    def __len__(self):
        return self.img_num


# =================
# get dataloader
# =================

def get_data_loaders(img_dir=None, noisy_image_dir=None, patch_size=256, batch_size=8, tform_op=None, noise_type='gaussian', noise_params={'sigma': 0.05}, shuffle=True, validation_split=None, status='train', num_workers=8, pin_memory=False, prefetch_factor=2, all2CPU=True):
    # dataset
    if status in ['train', 'test', 'valid']:
        if all2CPU:
            dataset = ImgNoiseDataset_all2CPU(
                img_dir, patch_size, tform_op, noise_type, noise_params)
        else:
            dataset = ImgNoiseDataset(
                img_dir, patch_size, tform_op, noise_type, noise_params)
    elif status in ['simuexp', 'realexp']:
        dataset = ImgNoiseDataset_Exp_all2CPU(
            noisy_image_dir, img_dir, patch_sz=patch_size, exp_mode=status)
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
    if status=='train':
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
    from utils.utils_noise_zzh import BasicNoiseModel, CMOS_Camera
    from utils.utils_image_zzh import augment_img

    img_dir = 'dataset/train_data/Kodak24/'
    save_dir = './outputs/tmp/test/'
