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
from srcs.utils.utils_eval_zzh import gpu_inference_time_est
from ptflops import get_model_complexity_info


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

    # calc MACs & Param. Num
    inputs_shape = [256, 256]
    macs, params = get_model_complexity_info(
        model=model, input_res=(3, *inputs_shape), verbose=False, print_per_layer_stat=False)
    logger.info(
        '='*40+'\n{:<30} {}'.format('Inputs resolution: ', inputs_shape))
    logger.info(
        '{:<30} {}'.format('Computational complexity: ', macs))
    logger.info('{:<30}  {}\n'.format(
        'Number of parameters: ', params)+'='*40)

    # DP
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)

    # load weight
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # instantiate loss and metrics
    criterion = {}
    if 'main_loss' in loaded_config.losses:
        criterion['main_loss'] = instantiate(
            loaded_config.main_loss, is_func=True)
    if 'loss2' in loaded_config.losses:
        criterion['loss2'] = instantiate(
            loaded_config.loss2, is_func=True)
    if 'loss3' in loaded_config.losses:
        criterion['loss3'] = instantiate(
            loaded_config.loss3, is_func=True)

    # metrics = [instantiate(met, is_func=True) for met in loaded_config.metrics]
    metrics = [instantiate(met) for met in config.metrics]

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
    if config.get('save_img', False):
        os.makedirs(config.outputs_dir+'/input')
        os.makedirs(config.outputs_dir+'/output')
        os.makedirs(config.outputs_dir+'/kernel')

    # inference time test
    input_shape = (1, 3, 256, 256)  # test image size
    gpu_inference_time_est(model, input_shape)

    # eval
    model.eval()
    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))
    time_start = time.time()
    with torch.no_grad():
        for i, (data_noisy, kernel, target) in enumerate(tqdm(test_data_loader, desc='Testing')):
            data_noisy, kernel, target = data_noisy.to(device), kernel.to(device), target.to(
                device)

            output, = model(data_noisy, kernel)

            # save some sample images
            if config.get('save_img', False):
                for k, (in_img, kernel_img, out_img, gt_img) in enumerate(zip(data_noisy, kernel, output, target)):
                    in_img = tensor2uint(in_img)
                    kernel_img = tensor2uint(kernel_img/torch.max(kernel_img))
                    out_img = tensor2uint(out_img)
                    gt_img = tensor2uint(gt_img)

                    imsave(
                        in_img, f'{config.outputs_dir}input/test{i+1:03d}_{k+1:03d}_input.png')
                    # imsave(
                    #     kernel_img, f'{config.outputs_dir}kernel/test{i+1:03d}_{k+1:03d}_kernel.png')
                    imsave(
                        out_img, f'{config.outputs_dir}output/test{i+1:03d}_{k+1:03d}_output.png')

                   # break  # save one image per batch

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
