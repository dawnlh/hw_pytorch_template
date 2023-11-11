import logging
import os
import numpy as np
import torch
import time
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from srcs.utils._util import instantiate
from srcs.utils.utils_image_zzh import tensor2uint, imsave_n
from srcs.utils.utils_eval_zzh import gpu_inference_time
from ptflops import get_model_complexity_info

def testing(config):
    test_worker(config)


def test_worker(config):
    ## logger & dir setting
    logger = logging.getLogger('test')
    os.makedirs(config.outputs_dir,exist_ok=True)

    ## prepare model & checkpoint for testing
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
    model = instantiate(loaded_config.arch)
    
    # load weight
    state_dict = checkpoint['state_dict']
    if len(loaded_config.gpus)>1:
        # preprocess DDP saved cpk
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # model = instantiate(config.arch)
    logger.info(model)

    ## calc MACs & Param. Num
    # define a `input_constructor` function to customized the inputs for ptflops
    def input_constructor(in_shape):
        # the parameter name must be consistent with that of the forword function
        input_kwargs = {'img': torch.randn(in_shape, dtype=torch.float32)}
        return input_kwargs
    # calc MACs & Param. Num
    inputs_shape = (1, 3, 256, 256)
    macs, params = get_model_complexity_info(
        model=model, input_res=inputs_shape, input_constructor=input_constructor, verbose=False, print_per_layer_stat=False)
    logger.info(
        '='*40+'\n{:<30} {}'.format('Inputs resolution: ', inputs_shape))
    logger.info(
        '{:<30} {}'.format('Computational complexity: ', macs))
    logger.info('{:<30}  {}\n'.format(
        'Number of parameters: ', params)+'='*40)

    ## DP
    if config.n_gpus > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(config.n_gpus)))

    # instantiate loss and metrics
    # criterion = instantiate(config.loss, is_func=True)

    metrics = [instantiate(met, is_func=False) for met in loaded_config.metrics]
    # metrics = [instantiate(met, is_func=True) for met in config.metrics]

    # setup data_loader instances
    test_data_loader = instantiate(config.test_data_loader)

    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = test(test_data_loader, model,
               device, metrics, config)
    logger.info(log)


def test(test_data_loader, model,  device, metrics, config):
    '''
    test step
    '''

    # init
    model = model.to(device)
    if config.get('save_img', False):
        os.makedirs(config.outputs_dir+'/images/')

    # inference time test
    input_shape = (1, 3, 256, 256)  # test image size
    gpu_inference_time(model, input_shape)

    # eval
    model.eval()
    total_metrics = torch.zeros(len(metrics), device=device)
    time_start = time.time()
    with torch.no_grad():
        for i, (img_noise, img_target, noise_params) in enumerate(tqdm(test_data_loader, desc='⏳ Testing')):
            img_noise, img_target = img_noise.to(device), img_target.to(device)

            # inference
            output = model(img_noise)

            # save image
            if config.get('save_img', False):
                for k, (in_img, out_img, gt_img) in enumerate(zip(img_noise, output, img_target)):
                    in_img = tensor2uint(in_img)
                    out_img = tensor2uint(out_img)
                    gt_img = tensor2uint(gt_img)
                    imgs = [in_img, out_img, gt_img]
                    imsave_n(
                        imgs, f'{config.outputs_dir}/images/test{i+1:03d}_{k+1:03d}.png')
                    # imsave_n(
                    #     [out_img], f'{config.outputs_dir}/images/test{i+1:03d}_{k+1:03d}_out.png')
                    
            # computing metrics on test set (if gt is available)
            if config.status != 'realexp':
                batch_size = img_noise.shape[0]
                for i, metric in enumerate(metrics):
                    total_metrics[i] += metric(output, img_target) * batch_size
    time_end = time.time()
    time_cost = time_end-time_start
    n_samples = len(test_data_loader.sampler)
    log = {'time/sample': time_cost/n_samples}
    if config.status!='realexp':
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
        })
    return log
