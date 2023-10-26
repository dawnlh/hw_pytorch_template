# Valid dataset Configuration

> Optional, only for conducting validation with specific dataset
> By default, validation dataset is part of training dataset
> Related to `congfig.trainer.assigned_valid`

## Hydra head statement

```yaml
# @package valid_data_loader
```

## Dataloader
```yaml
_target_: srcs.data_loader.basenet_data_loaders.get_data_loaders
gt_img_dir: ${hydra:runtime.cwd}/dataset/test_data/BSDS500_demo/ # valid dataset (str or list [str]) 
# noisy_image_dir: ${hydra:runtime.cwd}/dataset/real_data/BSDS500_demo/ # real_data (str or list [str]) 
batch_size: 1
patch_size: # default: full image
tform_op: ~
noise_type: camera # gaussian | camera
# noise_params: {'sigma': 0.05} # for gaussian noise
noise_params: {'sigma_beta': [0.01, 0.03], 'sigma_r': [0.5, 4], 'nd_factor': [2, 8], 'kc': 16} # for camera noise, kc: 4|8|16
status: 'valid'
shuffle: False
num_workers: ${num_workers}
pin_memory: True
prefetch_factor: 2
all2CPU: True # load all data to CPU
```