import logging
import os
import cv2
import torch
import hydra
import time
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from srcs.utils._util import instantiate
from srcs.utils.utils_image_kair import tensor2uint, imsave
from srcs.utils.utils_eval_zzh import gpu_inference_time

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
    ## load checkpoint
    logger.info('Loading checkpoint a: {} ...'.format(config.checkpoint_a))
    checkpoint_a = torch.load(config.checkpoint_a)
    if 'config' in checkpoint_a:
        loaded_config_a = OmegaConf.create(checkpoint_a['config'])
    else:
        loaded_config_a = config

    logger.info('Loading checkpoint b: {} ...'.format(config.checkpoint_b))
    checkpoint_b = torch.load(config.checkpoint_b)
    if 'config' in checkpoint_b:
        loaded_config_b = OmegaConf.create(checkpoint_b['config'])
    else:
        loaded_config_b = config

    ## instantiate model
    model_a = instantiate(loaded_config_a.arch_a)
    logger.info(model_a)
    if len(gpus) > 1:
        model_a = torch.nn.DataParallel(model_a, device_ids=gpus)

    model_b = instantiate(loaded_config_b.arch_b)
    logger.info(model_b)
    if len(gpus) > 1:
        model_b = torch.nn.DataParallel(model_b, device_ids=gpus)

    ## load weight
    state_dict_a = checkpoint_a['state_dict']
    model_a.load_state_dict(state_dict_a)

    state_dict_b = checkpoint_b['state_dict']
    model_b.load_state_dict(state_dict_b)

    # instantiate loss and metrics
    # get function handles of loss
    criterion_a = {}
    if 'gt_loss' in config.loss_a:
        criterion_a['gt_loss'] = instantiate(
            config.loss_a.gt_loss, is_func=True)
    if 'reblur_loss' in config.loss_a:
        criterion_a['reblur_loss'] = instantiate(
            config.loss_a.reblur_loss, is_func=True)
    criterion_b = {}
    if 'gt_loss' in config.loss_b:
        criterion_b['gt_loss'] = instantiate(
            config.loss_b.gt_loss, is_func=True)
    if 'reblur_loss' in config.loss_b:
        criterion_b['reblur_loss'] = instantiate(
            config.loss_b.reblur_loss, is_func=True)

    # get function handles of metrics
    metrics_a = [instantiate(met, is_func=True) for met in config['metrics_a']]
    metrics_b = [instantiate(met, is_func=True) for met in config['metrics_b']]

    # setup data_loader instances
    data_loader = instantiate(config.test_data_loader)

    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = test(model_a, model_b, metrics_a, metrics_b,
               data_loader, device,  config)
    logger.info(log)


def test(model_a, model_b, metrics_a, metrics_b,
         data_loader, device,  config):
    '''
    test step
    '''

    # init
    ce_code = config.get('ce_code', None)  # coding sequence
    model_a = model_a.to(device)
    model_b = model_b.to(device)
    model_a.eval()
    model_b.eval()
    total_metrics_a = torch.zeros(len(metrics_a))
    total_metrics_b = torch.zeros(len(metrics_b))
    time_cost = 0

    # test
    with torch.no_grad():
        for i, (target, data, kernel, sigma) in enumerate(tqdm(data_loader, desc='Testing')):
            data, target, kernel = data.to(
                device), target.to(device), kernel.to(device)
            # timer start
            time_start = time.time()

            # run model a - Knet_adaptivePool
            ce_code = list(ce_code)
            kernel_est = model_a(data, ce_code)

            # run model b
            output_ = model_b(data, kernel_est)
            output = output_[1]

            # timer end
            time_end = time.time()
            time_cost += time_end-time_start

            # save some sample images
            for k, (in_img, out_img, gt_img) in enumerate(zip(data, output, target)):
                in_img = tensor2uint(in_img)
                out_img = tensor2uint(out_img)
                gt_img = tensor2uint(gt_img)
                ker_est_img = tensor2uint(kernel_est/kernel_est.max())
                ker_gt_img = tensor2uint(kernel/kernel.max())
                imsave(
                    in_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_in_img.jpg')
                imsave(
                    out_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_out_img.jpg')
                imsave(
                    gt_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_gt_img.jpg')
                imsave(
                    ker_est_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_ker_est_img.jpg')
                imsave(
                    ker_gt_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_ker_gt_img.jpg')
                # break  # save one image per batch

            batch_size = data.shape[0]
            for i, metric in enumerate(metrics_a):
                total_metrics_a[i] += metric(output, target) * batch_size
            for i, metric in enumerate(metrics_b):
                total_metrics_b[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'time/sample': time_cost/n_samples}
    log.update({
        'model_a-'+met.__name__: total_metrics_a[i].item() / n_samples for i, met in enumerate(metrics_a)
    })
    log.update({
        'model_b-'+met.__name__: total_metrics_b[i].item() / n_samples for i, met in enumerate(metrics_b)
    })
    return log
