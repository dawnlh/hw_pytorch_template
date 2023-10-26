# Test/Exp dataset Configuration

## Hydra head statement

```yaml
# @package test_data_loader
```

## Dataloader
```yaml
_target_: srcs.data_loader.basenet_data_loaders.get_data_loaders
gt_img_dir: ${hydra:runtime.cwd}/dataset/test_data/DAVIS_demo/ # dir path of test gt (str or list [str]) 
# noisy_image_dir: ${hydra:runtime.cwd}/dataset/real_data/DAVIS_demo/ # dir path of test input (str or list [str]) 
batch_size: 4 # 1
patch_size: ~ # default: full image
tform_op: ~
noise_type: camera # gaussian | camera
# noise_params: {'sigma': 0.05} # for gaussian noise
noise_params: {'sigma_beta': [0.01, 0.03], 'sigma_r': [0.5, 4], 'nd_factor': [2, 8], 'kc': 16} # for camera noise, kc: 4|8|16
status: ${status}
shuffle: False
num_workers: ${num_workers}
pin_memory: True
prefetch_factor: 2
all2CPU: True # load all data to CPU
```
