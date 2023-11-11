# ======================================
# Trainning Engine: run Trainer for trainning
# ======================================
import torch
import platform
import os
from omegaconf import OmegaConf
from srcs.utils._util import instantiate, get_logger
import torch.distributed as dist

# training engine


def train_engine(trainer, gpus, config):
    if config.n_gpus > 1:
        # multiple GPUs
        torch.multiprocessing.spawn(
            multi_gpu_train_worker, nprocs=config.n_gpus, args=(trainer, gpus, config))
    else:
        # single gpu
        train_worker(trainer, config)

# training worker for single GPU


def train_worker(trainer, config):
    # setup logger
    logger = get_logger('train')
    # setup data_loader instances
    train_data_loader, valid_data_loader = instantiate(
        config.data_loader)

    # conduct test ater training
    if config.trainer.final_test:
        test_data_loader = instantiate(config.test_data_loader)
    else:
        test_data_loader = None

    # use assigned validation during training
    if config.trainer.assigned_valid:
        logger.info('📣 using assigned validation set')
        valid_data_loader = instantiate(config.valid_data_loader)

    if not valid_data_loader:
        logger.warning('⚠️ validation set  is empty =!')

    # build model & print its structure
    model = instantiate(config.arch)
    logger.info(model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(
        '='*40+f'\nTrainable parameters: {sum([p.numel() for p in trainable_params])/1e6:.2f} M\n'+'='*40)

    # get function handles of loss and metrics
    criterion = instantiate(config.loss, is_func=True)

    # metrics = [instantiate(met, is_func=True) for met in config['metrics']]
    metrics = [instantiate(met, is_func=False) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)
    trainer = trainer(model, config, criterion, metrics, optimizer, lr_scheduler,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader)
    trainer.train()


# training worker for multiple GPU
def multi_gpu_train_worker(rank, trainer, gpus, config):
    """
    Training with multiple GPUs

    Args:
        rank ([type]): [description]
        gpus ([type]): [description]
        config ([type]): [description]

    Raises:
        RuntimeError: [description]
    """
    # initialize training config
    config.local_rank = rank
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    if(platform.system() == 'Windows'):
        backend = 'gloo'
    elif(platform.system() == 'Linux'):
        backend = 'nccl'
    else:
        raise RuntimeError('Unknown Platform (Windows and Linux are supported')
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:34567',
        world_size=len(gpus),
        rank=rank)
    torch.cuda.set_device(rank)

    # start training processes
    train_worker(trainer, config)
