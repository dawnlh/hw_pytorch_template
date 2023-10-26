# Test Configuration

##  Hydra
```yaml
hydra:
    run:
      dir: ./outputs/${exp_name}/${status}/${now:%Y-%m-%d}_${now:%H-%M-%S}
    sweep:
      dir: ./outputs/${exp_name}/${status}/${now:%Y-%m-%d}_${now:%H-%M-%S}
      subdir: ${hydra.job.override_dirname}
```


## Dir
```yaml
exp_name: code_dev/basenet  # experiment name
tester_name: basenet_tester # tester name
status: test                # status: test|simuexp|realexp
checkpoint: ${hydra:runtime.cwd}/outputs/code_dev/basenet/train/2023-10-25_21-43-19/checkpoints/model_best.pth    # loading checkpoint (${hydra:runtime.cwd})
outputs_dir: outputs/       # output dir
```


## Runtime
```yaml
gpus: [4,6]         # GPU used, Warning: Only one GPU is supported for 'test' now
num_workers: 8      # number of cpu worker
save_img: true      # save the reconstructed images
```

## Architecture 
> optional, default to use `arch` in loaded config
```yaml
network_name: basenet
arch:
  _target_: srcs.model.basenet_model.basenet
  n_colors: 3
  nc: [32, 64, 128, 256]
```

## Metrics
```yaml
metrics:
  - _target_: srcs.metric.metric_iqa.IQA_Metric
    metric_name: psnr
  - _target_: srcs.metric.metric_iqa.IQA_Metric
    metric_name: ssim
  - _target_: srcs.metric.metric_iqa.IQA_Metric
    metric_name: lpips
```


## Dataset and others
```yaml
defaults:
  - data: basenet_test_data      # basenet_test_data | basenet_exp_data
  - override hydra/job_logging : custom # custom | colorlog
  - override hydra/hydra_logging: colorlog
  - _self_
```
