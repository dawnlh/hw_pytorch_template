# Train dataset Configuration

## Hydra head statement

```yaml
# @package data_loader
```

## Dataloader

```yaml
_target_: srcs.data_loader.basenet_data_loaders.get_data_loaders
gt_img_dir: ['${hydra:runtime.cwd}/dataset/train_data/Kodak24/'] # training dataset (str or list [str]) 
batch_size: ${batch_size}
patch_size: ${patch_size}
tform_op: 'all' # 'flip' | 'rotate' | 'reverse' | None - no image augment
noise_type: 'camera' # 'gaussian' | 'camera'
# noise_params: {'sigma': [0,0.05]} # for gaussian
noise_params: {'sigma_beta': [0.01, 0.03], 'sigma_r': [0.5, 4], 'nd_factor': [2, 8], 'kc': [4,16]} # for camera,exp
status: ${status}
shuffle: true
num_workers: ${num_workers}
validation_split: 0.05
pin_memory: False
prefetch_factor: 2
all2CPU: True # load all data to CPU
```