
# Train Configuration

## Hydra
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
exp_name: code_dev/basenet        # experiment name
status: train                     # status: train
# resume_conf: [epoch, optimizer] # resume configuration, default=[epoch, optimizer]
resume: ~                         # resuming checkpoint path (${hydra:runtime.cwd}/ )
checkpoint_dir: checkpoints/      # models saving dir (relative to $dir)
final_test_dir: final_test/       # final test result saving dir (relative to $dir)
log_dir: events/                  # log file save dir (relative to $dir)
```

## Runtime
```yaml
gpus: [0]                    # GPU used, list (None | -1 | empty -> all gpu)
num_workers: 4                 # number of cpu workers
trainer_name: basenet_trainer  # trainer name
trainer:
  epochs: 400                  # maximal training epochs
  limit_train_iters: ~         # maximal trainning batches: empty for all
  limit_valid_iters: ~         # maximal validation batches: empty for all
  monitor: min loss/valid      # monitor for early stop: max calc_psnr/valid | min loss/valid
  saving_top_k: 5              # save top k checkpoints (best checkpoints saved separately)
  early_stop: 100              # stop if no improvement in consecutive $early_stop epochs, 10
  logging_step: 500            # one log / $logging_step iteration
  tensorboard: true            # use tensorboard for training log
  log_weight: False            # log weight in tensorboard
  final_test: False            # do test after the training
  assigned_valid: False        # use assigned validation set
```

## Metrics
```yaml
metrics:
  - _target_: srcs.metric.metric_iqa.IQA_Metric
    metric_name: psnr
  - _target_: srcs.metric.metric_iqa.IQA_Metric
    metric_name: ssim
```

## Dataset, Network, Hyperparameters and others
```yaml
defaults:
  - data: basenet_train_data   # used datasets: [basenet_train_data,basenet_test_data,basenet_valid_data] 
  - network: basenet           # network
  - hparams: basenet_hyparams
  - override hydra/job_logging : custom # custom || colorlog
  - override hydra/hydra_logging: colorlog
  - _self_
```