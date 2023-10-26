# Network Configuration

## Hydra head statement

```yaml
# @package _global_
```

## Architecture

```yaml
network_name: basenet
arch:
  _target_: srcs.model.my_model.my_arch
  n_in: 3
```

## Loss function

### type1: compound loss

```yaml
loss: {'main_loss':1, 'loss2':0.5, 'loss3':0.5} # main_loss, loss2, loss3
main_loss:
  _target_: srcs.loss._pix_loss_func.weighted_loss
  loss_conf_dict: {'l1_loss':0.4, 'mse_loss':0.2, 'ssim_loss':0.2, 'tv_loss':0.2}
loss2:
  _target_: srcs.loss._pix_loss_func.weighted_loss
  loss_conf_dict: {'l1_loss':0.5, 'mse_loss':0.3, 'tv_loss':0.2}
loss3:
  _target_: srcs.loss._pix_loss_func.l1_loss
```

### type2: single loss

```yaml
loss:
  # loss without param
  _target_: srcs.loss._pix_loss_func.l1_loss 

  # loss with params
  # _target_: srcs.loss._pix_loss_cls.WeightedLoss 
  # loss_conf_dict: {'CharbonnierLoss':1.0, 'EdgeLoss':0.05, 'fftLoss':0.01} 
```

## Optimizer

```yaml
# Adam optimizer
optimizer:
  _target_: torch.optim.Adam
  lr: ${learning_rate}
  weight_decay: ${weight_decay}
  amsgrad: true

# Adan optimizer
  _target_: srcs.optimizer.adan.Adan
  lr: !!float 5e-4

```

## LR scheduler

```yaml
# StepLR
lr_scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: ${scheduler_step_size}
  gamma: ${scheduler_gamma}

# CosineAnnealingLR
lr_scheduler:
  _target_: srcs.scheduler._base_scheduler.getGradualWarmupScheduler
  multiplier: 1
  warmup_epochs: 2
  after_scheduler_conf:
    type: torch.optim.lr_scheduler.CosineAnnealingLR
    args:
      T_max: ${trainer.epochs}
      eta_min: 1e-6
```

