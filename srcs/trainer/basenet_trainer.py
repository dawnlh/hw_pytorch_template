from tqdm import tqdm
import torch
import time
import torch.distributed as dist
from torchvision.utils import make_grid
from ._base import BaseTrainer
from srcs.utils._util import collect

from srcs.utils.utils_image_zzh import tensor2uint, imsave_n

# ======================================
# Trainer
# ======================================


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, config, criterion, metrics, optimizer, lr_scheduler=None, train_data_loader=None, valid_data_loader=None, test_data_loader=None):
        super().__init__(model, config, criterion, metrics, optimizer, lr_scheduler,
                         train_data_loader, valid_data_loader, test_data_loader)
        
        # self.loss = self.config['losses']
        self.grad_clip = 0.5  # optimizer gradient clip value

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

    def _after_iter(self, epoch, batch_idx, phase, loss, metrics, image_tensors: dict):
        # hook after iter
        self.writer.set_step(
            (epoch - 1) * getattr(self, f'limit_{phase}_iters') + batch_idx, speed_chk=f'{phase}')

        loss_v = loss.item() if self.config.n_gpus == 1 else collect(loss)
        getattr(self, f'{phase}_metrics').update('loss', loss_v)

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            getattr(self, f'{phase}_metrics').update(
                k, v)  # `v` is a torch tensor

        for k, v in image_tensors.items():
            self.writer.add_image(
                f'{phase}/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (img_noise, img_target, noise_params) in enumerate(self.data_loader):
            img_noise, img_target = img_noise.to(
                self.device), img_target.to(self.device)

            # forward
            output = self.model(img_noise)

            # loss calc
            loss = self.criterion(output, img_target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            # clip gradient and update
            self.clip_gradient(self.optimizer, self.grad_clip)
            self.optimizer.step()

            # iter record
            if batch_idx % self.logging_step == 0 or (batch_idx+1) == self.limit_train_iters:
                # iter metrics
                iter_metrics = {}
                for met in self.metrics:
                    if self.config.n_gpus > 1:
                        # average metric between processes
                        metric_v = collect(met(output, img_target))
                    else:
                        # print(output.shape, img_target.shape)
                        metric_v = met(output, img_target)
                    iter_metrics.update({met.__name__: metric_v})

                # iter images
                image_tensors = {
                    'input': img_noise[0:4, ...], 'img_target': img_target[0:4, ...], 'output': output[0:4, ...]}

                # aftet iter hook
                self._after_iter(epoch, batch_idx, 'train',
                                 loss, iter_metrics, {})  # don't save images in every iter to save space
                # iter log
                self.logger.info(
                    f'Train Epoch: {epoch:03d} {self._progress(batch_idx)} Loss: {loss:.6f} Lr: {self.optimizer.param_groups[0]["lr"]:.3e}')

            # reach maximum training iters, endding epoch
            if (batch_idx+1) == self.limit_train_iters:
                # save demo images to tensorboard after trainig epoch
                self.writer.set_step(epoch)
                for k, v in image_tensors.items():
                    self.writer.add_image(
                        f'train/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
                break  # endding epoch

        # epoch log
        log = self.train_metrics.result()

        # valid after every epoch
        if self.valid_data_loader:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

        # learning rate update
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # add result metrics on entire epoch to tensorboard
        self.writer.set_step(epoch)
        for k, v in log.items():
            self.writer.add_scalar(k + '/epoch', v)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            # show ce code update only when optimized
            for batch_idx, (img_noise, img_target, noise_params) in enumerate(self.valid_data_loader):
                img_noise, img_target = img_noise.to(
                    self.device), img_target.to(self.device)

                # forward
                output = self.model(img_noise)

                # loss calc
                loss = self.criterion(output, img_target)

                # iter record
                # iter metrics
                iter_metrics = {}
                for met in self.metrics:
                    if self.config.n_gpus > 1:
                        # average metric between processes
                        metric_v = collect(met(output, img_target))
                    else:
                        # print(output.shape, img_target.shape)
                        metric_v = met(output, img_target)
                    iter_metrics.update({met.__name__: metric_v})

                # iter images
                image_tensors = {
                    'input': img_noise[0:4, ...], 'img_target': img_target[0:4, ...], 'output': output[0:4, ...]}

                # aftet iter hook
                self._after_iter(epoch, batch_idx, 'valid',
                                 loss, iter_metrics, {})  # don't save images in every iter to save space

                # reach maximum validation iters, endding epoch
                if (batch_idx+1) == self.limit_valid_iters:
                    # save demo images to tensorboard after valid epoch
                    self.writer.set_step(epoch)
                    for k, v in image_tensors.items():
                        self.writer.add_image(
                            f'valid/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
                    break  # endding epoch

        # add histogram of model parameters to the tensorboard
        if self.log_weight:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _test_epoch(self):
        """
        Final test logic after the training (! test the latest checkpoint)

        :param epoch: Current epoch number
        """
        self.model.eval()
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metrics))
        time_start = time.time()

        with torch.no_grad():
            for batch_idx, (img_noise, img_target, noise_params) in enumerate(tqdm(self.test_data_loader)):
                img_noise, img_target = img_noise.to(
                    self.device), img_target.to(self.device)

                # forward
                output = self.model(img_noise)

                # save some sample images
                for k, (in_img, out_img, gt_img) in enumerate(zip(img_noise, output, img_target)):
                    in_img = tensor2uint(in_img)
                    out_img = tensor2uint(out_img)
                    gt_img = tensor2uint(gt_img)
                    imgs = [in_img, out_img, gt_img]
                    imsave_n(imgs, f'{self.final_test_dir}/test{i+1:03d}_{k+1:03d}_img.png')
                    # imsave_n([out_img], f'{self.final_test_dir}/test{i+1:03d}_{k+1:03d}_out_img.png')
                    
                # computing loss, metrics on test set
                loss = self.criterion(output, img_target)
                batch_size = img_noise.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metrics):
                    total_metrics[i] += metric(output, img_target) * batch_size
        time_end = time.time()
        time_cost = time_end-time_start
        n_samples = len(self.test_data_loader.sampler)
        log = {'loss': total_loss / n_samples,
               'time/sample': time_cost/n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metrics)
        })
        self.logger.info('-'*70+'\n Final test result: '+str(log))

