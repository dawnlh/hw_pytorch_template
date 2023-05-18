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
from srcs.utils.utils_eval_zzh import gpu_inference_time
from ptflops import get_model_complexity_info

def testing(gpus, config):
    test_worker(gpus, config)


def test_worker(gpus, config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    # logger & dir setting
    logger = logging.getLogger('test')
    os.makedirs(config.outputs_dir,exist_ok=True)

    # prepare model & checkpoint for testing
    # load checkpoint
    logger.info(f"📥 Loading checkpoint: {config.checkpoint} ...")
    checkpoint = torch.load(config.checkpoint)
    logger.info(f"💡 Checkpoint loaded: epoch {checkpoint['epoch']}.")

    # select config file
    if 'config' in checkpoint:
        loaded_config = OmegaConf.create(checkpoint['config'])
    else:
        loaded_config = config

    # instantiate model
    # model = instantiate(loaded_config.arch)
    model = instantiate(config.arch)
    logger.info(model)

    # calc MACs & Param. Num
    inputs_shape = (3, 256, 256)

    def input_constructor(in_shape):
        # define a `input_constructor` function to customized the inputs for ptflops
        # the
        input_kwargs = {'frames': torch.randn(in_shape, device="cuda:0")}
        return input_kwargs
    macs, params = get_model_complexity_info(
        model=model, input_res=inputs_shape, input_constructor=input_constructor, verbose=False, print_per_layer_stat=False)
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
    if 'input_denoise_loss' in loaded_config.losses:
        criterion['input_denoise_loss'] = instantiate(
            loaded_config.input_denoise_loss, is_func=True)
    if 'forward_conv_loss' in loaded_config.losses:
        criterion['forward_conv_loss'] = instantiate(
            loaded_config.forward_conv_loss, is_func=True)

    # metrics = [instantiate(met, is_func=True) for met in loaded_config.metrics]
    metrics = [instantiate(met, is_func=True) for met in config.metrics]

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
        os.makedirs(config.outputs_dir+'/data-denoise')

    # inference time test
    input_shape = (1, 3, 256, 256)  # test image size
    gpu_inference_time(model, input_shape)

    # eval
    model.eval()
    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))
    time_start = time.time()
    with torch.no_grad():
        for i, (data_noisy, kernel, target, data) in enumerate(tqdm(test_data_loader, desc='⏳ Testing')):
            data_noisy, kernel, target = data_noisy.to(device), kernel.to(device), target.to(
                device)

            output, data_denoise = model(data_noisy, kernel)

            # save some sample images
            output = output[1]  # zzh: deblured image

            if config.get('save_img', False):
                for k, (in_img, kernel_img, out_img, gt_img) in enumerate(zip(data_noisy, kernel, output, target)):
                    in_img = tensor2uint(in_img)
                    kernel_img = tensor2uint(kernel_img/torch.max(kernel_img))
                    out_img = tensor2uint(out_img)
                    gt_img = tensor2uint(gt_img)
                    data_denoise_img = tensor2uint(data_denoise)

                    # crop for symmetric padding
                    if config['status'] == 'realexp':
                        H, W = in_img.shape[0]//2, in_img.shape[1]//2
                        h = np.int32(H/2)
                        w = np.int32(W/2)
                        in_img = in_img[h:h+H, w:w+W]
                        out_img = out_img[h:h+H, w:w+W]
                        data_denoise_img = data_denoise_img[h:h+H, w:w+W]

                    imsave(
                        in_img, f'{config.outputs_dir}input/test{i+1:03d}_{k+1:03d}_input.png')
                    # imsave(
                    #     kernel_img, f'{config.outputs_dir}kernel/test{i+1:03d}_{k+1:03d}_kernel.png')
                    imsave(
                        out_img, f'{config.outputs_dir}output/test{i+1:03d}_{k+1:03d}_output.png')
                    # imsave(
                    #     gt_img, f'{config.outputs_dir}test{i+1:03d}_{k+1:03d}_gt.png')
                    imsave(
                        data_denoise_img, f'{config.outputs_dir}data-denoise/test{i+1:03d}_{k+1:03d}_data-denoise.png')
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
