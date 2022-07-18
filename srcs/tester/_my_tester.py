import logging
import os
import cv2
import numpy as np
import torch
import hydra
import time
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from srcs.utils._util import instantiate
from srcs.utils.utils_image_kair import tensor2uint, imsave


def testing(gpus, config):
    test_worker(gpus, config)


def test_worker(gpus, config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    # logger & dir setting
    logger = logging.getLogger('test')
    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)

    # prepare model & checkpoint for testing
    # load checkpoint
    logger.info('Loading checkpoint: {} ...'.format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint)
    if 'config' in checkpoint:
        loaded_config = OmegaConf.create(checkpoint['config'])
    else:
        loaded_config = config

    # instantiate model
    model = instantiate(loaded_config.arch)
    logger.info(model)
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)

    # load weight
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # instantiate loss and metrics
    criterion = {}
    if 'main_loss' in loaded_config.loss:
        criterion['main_loss'] = instantiate(
            loaded_config.main_loss, is_func=True)
    if 'loss2' in loaded_config.loss:
        criterion['loss2'] = instantiate(
            loaded_config.loss2, is_func=True)
    if 'loss3' in loaded_config.loss:
        criterion['loss3'] = instantiate(
            loaded_config.loss3, is_func=True)

    metrics = [instantiate(met, is_func=True) for met in loaded_config.metrics]

    # setup data_loader instances
    test_data_loader = instantiate(config.test_data_loader)

    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = test(test_data_loader, model,
               device, criterion, metrics, config)
    logger.info(log)


def test(test_data_loader, model,  device, criterion, metrics, config):
    '''
    test step
    '''

    # init
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))
    time_start = time.time()
    with torch.no_grad():
        for i, (data_noisy, kernel, target) in enumerate(tqdm(test_data_loader, desc='Testing')):
            data_noisy, kernel, target = data_noisy.to(device), kernel.to(device), target.to(
                device)

            output, = model(data_noisy, kernel)


            # save sample images here
            for k, (in_img, kernel_img, out_img, gt_img) in enumerate(zip(data_noisy, kernel, output, target)):
                in_img = tensor2uint(in_img)
                kernel_img = tensor2uint(kernel_img/torch.max(kernel_img))
                out_img = tensor2uint(out_img)
                gt_img = tensor2uint(gt_img)

                imsave(
                    in_img, f'{config.outputs_dir}test{i+1:03d}_{k+1:03d}_input.png')
                # imsave(
                #     kernel_img, f'{config.outputs_dir}test{i+1:03d}_{k+1:03d}_kernel.png')
                imsave(
                    out_img, f'{config.outputs_dir}test{i+1:03d}_{k+1:03d}_output.png')
                break  # save one image per batch

            # computing loss, metrics on test set
            loss = criterion['main_loss'](output, target)
            batch_size = data_noisy.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(output, target) * batch_size
    time_end = time.time()
    time_cost = time_end-time_start
    n_samples = len(test_data_loader.sampler)
    log = {'loss': total_loss / n_samples,
           'time/sample': time_cost/n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    return log
