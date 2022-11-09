import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.utils import make_grid
from numpy import inf
import platform
import time
import os
from shutil import copyfile
from pathlib import Path
from omegaconf import OmegaConf
from ._base2 import BaseTrainer2
from srcs.utils._util import inf_loop, collect, instantiate, get_logger
from srcs.utils.utils_patch_proc import window_partitionx, window_reversex
from srcs.logger import BatchMetrics, TensorboardWriter, EpochMetrics
from srcs.utils._util import write_conf, is_master
from srcs.utils.utils_deblur_zzh import pad4conv, img_blur_torch
from ptflops import get_model_complexity_info

# ======================================
# Trainer: modify '_train_epoch'
# ======================================


class Trainer(BaseTrainer2):
    """
    Trainer class
    """

    def __init__(self, model_a, model_b, criterion_a, criterion_b, optimizer_a, optimizer_b, lr_scheduler_a, lr_scheduler_b, metrics_a, metrics_b, config, data_loader, valid_data_loader):
        super().__init__(model_a, model_b, criterion_a, criterion_b, optimizer_a, optimizer_b,
                         lr_scheduler_a, lr_scheduler_b, metrics_a, metrics_b, config, data_loader, valid_data_loader)

        self.losses_a = self.config['loss_a']['weights']
        self.losses_b = self.config['loss_b']['weights']
        self.loss_all = self.config['loss_all']

        self.init_epochs = self.config['trainer'].get(
            'init_epochs', {'a': 0, 'b': 0})  # init epoch
        self.grad_clip = 0.5  # optimizer gradient clip value
        self.win_size = 80  # slicing window size, psf size
        self.n_levels = 2  # model scale levels
        self.scales = [0.5, 1]  # model scale
        self.ce_code = config.get('ce_code', None)  # coding sequence

    def clip_gradient(self, optimizer, grad_clip=0.5):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def _after_iter(self, epoch, batch_idx, phase, model_id, loss, iter_metrics, image_tensors: dict):
        self.writer.set_step(
            (epoch - 1) * getattr(self, f'limit_{phase}_batches') + batch_idx, speed_chk=f'{phase}_{model_id}')

        loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
        getattr(self, f'{phase}_metrics_{model_id}').update('loss', loss_v)

        for k, v in iter_metrics.items():
            getattr(self, f'{phase}_metrics_{model_id}').update(k, v)

        for k, v in image_tensors.items():
            self.writer.add_image(
                f'{phase}_{model_id}/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))

    def _calc_loss_a(self, kernel_est, kernel, blur, sharp):
        loss = 0
        # gt_loss
        if 'gt_loss' in self.losses_a:
            gt_loss = self.criterion_a['gt_loss'](kernel_est, kernel)
            loss += self.losses_a['gt_loss']*gt_loss
        # reblur_loss
        if 'reblur_loss' in self.losses_a:
            # sharp image circular padding
            reblur = img_blur_torch(sharp, kernel_est)
            reblur_loss = self.criterion_b['reblur_loss'](
                reblur, blur)
            loss += self.losses_b['reblur_loss'] * reblur_loss
        return loss

    def _calc_loss_b(self, deblur_, kernel, blur, sharp):
        loss = 0
        # gt_loss
        if 'gt_loss' in self.losses_b:
            gt_loss = 0
            for level in range(self.n_levels):
                scale = self.scales[level]
                n, c, h, w = sharp.shape
                hi = int(round(h * scale))
                wi = int(round(w * scale))
                sharp_level = F.interpolate(
                    sharp, (hi, wi), mode='bilinear')
                gt_loss += self.criterion_b['gt_loss'](
                    deblur_[level], sharp_level)
            loss += self.losses_b['gt_loss']*gt_loss

        # reblur_loss
        if 'reblur_loss' in self.losses_b:
            # sharp image circular padding
            reblur = img_blur_torch(deblur_[1], kernel)
            reblur_loss = self.criterion_b['reblur_loss'](
                reblur, blur)
            loss += self.losses_b['reblur_loss'] * reblur_loss
        return loss

    def _train_epoch_a(self, epoch):
        """
        Training logic of model_a for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_a.train()
        # self.model_b.eval()
        # self.model_b.requires_grad = False  # fix model_b
        self.train_metrics_a.reset()

        for batch_idx, (target, data, kernel, sigma) in enumerate(self.data_loader):
            data, target, kernel = data.to(self.device), target.to(
                self.device), kernel.to(self.device)
            ce_code = torch.tensor(
                list(self.ce_code), dtype=torch.float, device=self.device).repeat(kernel.shape[0], 1)

            # run model - Knet_adaptivePool
            kernel_est = self.model_a(data, ce_code)

            # loss calc
            loss = self._calc_loss_a(kernel_est, kernel, data, target)

            # update
            self.optimizer_a.zero_grad()
            loss.backward()
            self.clip_gradient(self.optimizer_a, self.grad_clip)
            self.optimizer_a.step()

            # iter log
            if batch_idx % self.logging_step == 0 or batch_idx == self.limit_train_batches:
                iter_metrics = {}
                for met in self.metric_ftns_a:
                    metric_v = met(kernel_est, kernel)
                    iter_metrics.update({met.__name__: metric_v})

                image_tensors = {'input': data,
                                 'kernel': kernel, 'kernel_est': kernel_est}
                self._after_iter(epoch, batch_idx, 'train', 'a',
                                 loss, iter_metrics, image_tensors)

                self.logger.info(
                    f'Train Epoch (Net_a): {epoch} {self._progress(batch_idx)} Loss: {loss:.6f} Lr: {self.optimizer_a.param_groups[0]["lr"]:.3e}')

            if batch_idx == self.limit_train_batches:
                break

        # epoch log
        log = self.train_metrics_a.result()

        #  val
        if self.valid_data_loader is not None:
            val_log = self._valid_epoch_a(epoch)
        log.update(**val_log)

        # add result metrics on entire epoch to tensorboard
        # self.writer.set_step(epoch)
        # for k, v in log.items():
        #     self.writer.add_scalar(k + '/epoch', v)

        # update scheduler
        if self.lr_scheduler_a is not None:
            self.lr_scheduler_a.step()

        return log

    def _valid_epoch_a(self, epoch):
        """
        Validate of model_a  after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model_a.eval()
        self.valid_metrics_a.reset()
        with torch.no_grad():
            for batch_idx, (target, data, kernel, sigma) in enumerate(self.valid_data_loader):
                data, target, kernel = data.to(self.device), target.to(
                    self.device), kernel.to(self.device)
                ce_code = torch.tensor(
                    list(self.ce_code), dtype=torch.float, device=self.device).repeat(kernel.shape[0], 1)
                # # run model - Knet_psf
                # N, C, H, W = data.shape
                # data_, _ = window_partitionx(
                #     data, self.win_size, keep_edge=False)
                # ce_code = list(self.ce_code)
                # kernel_est_ = self.model_a(data_, ce_code)
                # kernel_est_ = torch.reshape(
                #     kernel_est_, (N, -1, self.win_size, self.win_size))
                # kernel_est = torch.mean(kernel_est_, dim=1, keepdim=True)
                # # normalize to sum=1
                # kernel_est = kernel_est / \
                #     torch.sum(kernel_est, dim=(2, 3), keepdim=True)

                # run model - Knet_adaptivePool
                kernel_est = self.model_a(data, ce_code)

                # loss calc
                loss = self._calc_loss_a(kernel_est, kernel, data, target)

                # iter log
                iter_metrics = {}
                for met in self.metric_ftns_a:
                    metric_v = met(kernel_est, kernel)
                    iter_metrics.update({met.__name__: metric_v})
                image_tensors = {'input': data,
                                 'kernel': kernel, 'kernel_est': kernel_est}
                self._after_iter(epoch, batch_idx, 'valid', 'a',
                                 loss, iter_metrics, image_tensors)

                if batch_idx == self.limit_valid_batches:
                    break

        # add histogram of model parameters to the tensorboard
        for name, p in self.model_a.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics_a.result()

    def _train_epoch_b(self, epoch):
        """
        Training logic of model_b for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_b.train()
        self.model_a.eval()
        self.model_a.requires_grad = False  # fix model_a

        self.train_metrics_b.reset()

        for batch_idx, (target, data, kernel, sigma) in enumerate(self.data_loader):
            data, target, kernel = data.to(self.device), target.to(
                self.device), kernel.to(self.device)
            ce_code = torch.tensor(
                list(self.ce_code), dtype=torch.float, device=self.device).repeat(kernel.shape[0], 1)
            # init
            self.optimizer_b.zero_grad()

            # run model
            if epoch <= self.init_epochs['b']:
                kernel_ = kernel  # use gt kernel
            else:
                kernel_est = self.model_a(data, ce_code)
                kernel_ = kernel_est  # use estimated kernel
            output_ = self.model_b(data, kernel_)
            output = output_[1]

            # loss calc
            loss = self._calc_loss_b(output_, kernel_, data, target)

            # update
            loss.backward()
            self.clip_gradient(self.optimizer_b, self.grad_clip)
            self.optimizer_b.step()

            # iter log
            if batch_idx % self.logging_step == 0 or batch_idx == self.limit_train_batches:
                iter_metrics = {}
                for met in self.metric_ftns_b:
                    metric_v = met(output, target)
                    iter_metrics.update({met.__name__: metric_v})

                image_tensors = {'input': data, 'kernel_gt': kernel, 'kernel_use': kernel_,
                                 'target': target, 'output': output}
                self._after_iter(epoch, batch_idx, 'train', 'b',
                                 loss, iter_metrics, image_tensors)
                self.logger.info(
                    f'Train Epoch (Net_b): {epoch} {self._progress(batch_idx)} Loss: {loss:.6f} Lr: {self.optimizer_b.param_groups[0]["lr"]:.3e}')

            if batch_idx == self.limit_train_batches:
                break

        # epoch log
        log = self.train_metrics_b.result()

        #  val
        if self.valid_data_loader is not None:
            val_log = self._valid_epoch_b(epoch)
        log.update(**val_log)

        # add result metrics on entire epoch to tensorboard
        # self.writer.set_step(epoch)
        # for k, v in log.items():
        #     self.writer.add_scalar(k + '/epoch', v)

        # update scheduler
        if self.lr_scheduler_b is not None:
            self.lr_scheduler_b.step()

        return log

    def _valid_epoch_b(self, epoch):
        """
        Validate of model_b  after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model_a.eval()
        self.model_b.eval()
        self.valid_metrics_b.reset()
        with torch.no_grad():
            for batch_idx, (target, data, kernel, sigma) in enumerate(self.valid_data_loader):
                data, target, kernel = data.to(self.device), target.to(
                    self.device), kernel.to(self.device)
                ce_code = torch.tensor(
                    list(self.ce_code), dtype=torch.float, device=self.device).repeat(kernel.shape[0], 1)

                # run model
                if epoch <= self.init_epochs['b']:
                    kernel_ = kernel  # use gt kernel
                else:
                    kernel_est = self.model_a(data, ce_code)
                    kernel_ = kernel_est  # use estimated kernel
                output_ = self.model_b(data, kernel_)
                output = output_[1]

                # loss calc
                loss = self._calc_loss_b(output_, kernel_, data, target)

                # iter log
                iter_metrics = {}
                for met in self.metric_ftns_b:
                    metric_v = met(output, target)
                    iter_metrics.update({met.__name__: metric_v})
                image_tensors = {'input': data, 'kernel_gt': kernel,
                                 'kernel_use': kernel_, 'target': target, 'output': output}
                self._after_iter(epoch, batch_idx, 'valid', 'b',
                                 loss, iter_metrics, image_tensors)

                if batch_idx == self.limit_valid_batches:
                    break

        # add histogram of model parameters to the tensorboard
        for name, p in self.model_b.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics_b.result()

    def _train_epoch_ab(self, epoch_a, epoch_b, *args, **kwargs):
        """
        Training logic of end-t0-end model_a & model_b for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_a.train()
        self.model_b.train()
        self.train_metrics_a.reset()
        self.train_metrics_b.reset()

        for batch_idx, (target, data, kernel, sigma) in enumerate(self.data_loader):
            data, target, kernel = data.to(self.device), target.to(
                self.device), kernel.to(self.device)
            ce_code = torch.tensor(
                list(self.ce_code), dtype=torch.float, device=self.device).repeat(kernel.shape[0], 1)
            # init
            self.optimizer_a.zero_grad()
            self.optimizer_b.zero_grad()

            # run model a - Knet_adaptivePool
            kernel_est = self.model_a(data, ce_code)

            # run model b
            output_ = self.model_b(data, kernel_est)
            output = output_[1]

            # calc loss a
            loss_a = self._calc_loss_a(kernel_est, kernel, data, target)
            # calc loss b
            loss_b = self._calc_loss_b(output_, kernel, data, target)

            # sum loss
            loss = loss_a*self.loss_all['loss_a'] + \
                loss_b*self.loss_all['loss_b']

            # update
            loss.backward()
            self.clip_gradient(self.optimizer_a, self.grad_clip)
            self.clip_gradient(self.optimizer_b, self.grad_clip)
            self.optimizer_a.step()
            self.optimizer_b.step()

            # iter log
            if batch_idx % self.logging_step == 0 or batch_idx == self.limit_train_batches:
                # e2e
                self.logger.info(
                    f"Train Epoch (E2E): {kwargs['n_idx']+1}/{kwargs['n']} {self._progress(batch_idx)} Total Loss: {loss:.6f} ")

                # model_a
                iter_metrics = {}
                for met in self.metric_ftns_a:
                    metric_v = met(kernel_est, kernel)
                    iter_metrics.update({met.__name__: metric_v})

                image_tensors = {'input': data,
                                 'kernel': kernel, 'kernel_est': kernel_est}
                self._after_iter(epoch_a, batch_idx, 'train', 'a',
                                 loss_a, iter_metrics, image_tensors)
                self.logger.info(
                    f'---> Net_a: Epoch: {epoch_a:03d} Loss: {loss_a:.6f} Lr: {self.optimizer_a.param_groups[0]["lr"]:.3e}')

                # model_b
                iter_metrics = {}
                for met in self.metric_ftns_b:
                    metric_v = met(output, target)
                    iter_metrics.update({met.__name__: metric_v})

                image_tensors = {'input': data, 'kernel': kernel,
                                 'target': target, 'output': output}
                self._after_iter(epoch_b, batch_idx, 'train', 'b',
                                 loss_b, iter_metrics, image_tensors)
                self.logger.info(
                    f'---> Net_b: Epoch: {epoch_b:03d} Loss: {loss_b:.6f} Lr: {self.optimizer_b.param_groups[0]["lr"]:.3e}')

            if batch_idx == self.limit_train_batches:
                break

        # epoch log
        log_a = self.train_metrics_a.result()
        log_b = self.train_metrics_b.result()

        # val
        if self.valid_data_loader is not None:
            val_log_a, val_log_b = self._valid_epoch_ab(epoch_a, epoch_b)
        log_a.update(**val_log_a)
        log_b.update(**val_log_b)

        # write epoch log
        # self.writer.set_step(epoch_a)
        # for k, v in log_a.items():
        #     self.writer.add_scalar(k + '/epoch', v)
        # self.writer.set_step(epoch_b)
        # for k, v in log_b.items():
        #     self.writer.add_scalar(k + '/epoch', v)

        # update scheduler
        if self.lr_scheduler_a is not None:
            self.lr_scheduler_a.step()

        if self.lr_scheduler_b is not None:
            self.lr_scheduler_b.step()

        return log_a, log_b

    def _valid_epoch_ab(self, epoch_a, epoch_b):
        """
        Validate of model_a & model_b  after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model_a.eval()
        self.model_b.eval()
        self.valid_metrics_a.reset()
        self.valid_metrics_b.reset()
        with torch.no_grad():
            for batch_idx, (target, data, kernel, sigma) in enumerate(self.valid_data_loader):
                data, target, kernel = data.to(self.device), target.to(
                    self.device), kernel.to(self.device)
                ce_code = torch.tensor(
                    list(self.ce_code), dtype=torch.float, device=self.device).repeat(kernel.shape[0], 1)
                # run model a - Knet_adaptivePool
                kernel_est = self.model_a(data, ce_code)

                # run model b
                output_ = self.model_b(data, kernel_est)
                output = output_[1]

                # calc loss a
                loss_a = self._calc_loss_a(kernel_est, kernel, data, target)
                # calc loss b
                loss_b = self._calc_loss_b(output_, kernel, data, target)
                # sum loss
                loss = loss_a*self.loss_all['loss_a'] + \
                    loss_b*self.loss_all['loss_b']

                # iter log
                # model_a
                iter_metrics = {}
                for met in self.metric_ftns_a:
                    metric_v = met(kernel_est, kernel)
                    iter_metrics.update({met.__name__: metric_v})
                image_tensors = {'input': data,
                                 'kernel': kernel, 'kernel_est': kernel_est}
                self._after_iter(epoch_a, batch_idx, 'valid', 'a',
                                 loss, iter_metrics, image_tensors)

                # model_b
                iter_metrics = {}
                for met in self.metric_ftns_b:
                    metric_v = met(output, target)
                    iter_metrics.update({met.__name__: metric_v})
                image_tensors = {'input': data, 'kernel': kernel,
                                 'target': target, 'output': output}
                self._after_iter(epoch_b, batch_idx, 'valid', 'b',
                                 loss, iter_metrics, image_tensors)

                if batch_idx == self.limit_valid_batches:
                    break

        # add histogram of model parameters to the tensorboard
        for name, p in self.model_a.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        for name, p in self.model_b.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics_a.result(), self.valid_metrics_b.result()

# ======================================
# Trainning: run Trainer for trainning
# ======================================


def trainning(gpus, config):
    # enable access to non-existing keys
    OmegaConf.set_struct(config, False)
    n_gpu = len(gpus)
    config.n_gpu = n_gpu
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    if n_gpu > 1:
        torch.multiprocessing.spawn(
            multi_gpu_train_worker, nprocs=n_gpu, args=(gpus, config))
    else:
        train_worker(config)


def train_worker(config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    logger = get_logger('train')
    # setup data_loader instances
    data_loader, valid_data_loader = instantiate(config.data_loader)

    # build model. print it's structure and # trainable params.
    model_a = instantiate(config.arch_a)
    trainable_params = filter(lambda p: p.requires_grad, model_a.parameters())
    logger.info(model_a)
    logger.info(
        f'Trainable parameters for KNet: {sum([p.numel() for p in trainable_params])}')

    model_b = instantiate(config.arch_b)
    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    logger.info(model)
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # logger.info(
    #     f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')
    macs, params = get_model_complexity_info(
        model=model, input_res=(3, config.patch_size, config.patch_size), verbose=False, print_per_layer_stat=False)
    logger.info(
        '='*40+'\n{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logger.info('{:<30}  {:<8}\n'.format(
        'Number of parameters: ', params)+'='*40)

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

    # build optimizer, learning rate scheduler.
    optimizer_a = instantiate(config.optimizer_a, model_a.parameters())
    lr_scheduler_a = instantiate(config.lr_scheduler_a, optimizer_a)
    optimizer_b = instantiate(config.optimizer_b, model_b.parameters())
    lr_scheduler_b = instantiate(config.lr_scheduler_b, optimizer_b)

    # get Trainer
    trainer = Trainer(model_a, model_b,
                      criterion_a, criterion_b,
                      optimizer_a, optimizer_b,
                      lr_scheduler_a, lr_scheduler_b,
                      metrics_a, metrics_b,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader)
    trainer.train()


def multi_gpu_train_worker(rank, gpus, config):
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
    torch.cuda.set_device(gpus[rank])

    # start training processes
    train_worker(config)
