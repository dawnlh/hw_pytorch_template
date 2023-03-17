import os
import time
import torch
import torch.distributed as dist
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel
from abc import abstractmethod, ABCMeta
from pathlib import Path
from shutil import copyfile
from numpy import inf
from srcs.utils._util import write_conf, is_master, get_logger, collect
from srcs.logger import TensorboardWriter, EpochMetrics, BatchMetrics
from os.path import join as opj

class BaseTrainer2(metaclass=ABCMeta):
    """
    Base class for all trainers with two models
    """
    pass

    def __init__(self, model_a, model_b, criterion_a, criterion_b, optimizer_a, optimizer_b, lr_scheduler_a, lr_scheduler_b, metrics_a, metrics_b, config, data_loader=None, valid_data_loader=None):
        self.config = config
        self.logger = get_logger('trainer')
        self.device = config.local_rank if config.n_gpu > 1 else 0
        self.model_a = model_a.to(self.device)
        self.model_b = model_b.to(self.device)

        self.checkpt_dir = Path(self.config.checkpoint_dir)
        log_dir = Path(self.config.log_dir)

        cfg_trainer = config['trainer']
        self.workflow = cfg_trainer['workflow']
        self.logging_step = cfg_trainer.get('logging_step', 100)
        self.monitor = cfg_trainer.get('monitor', 'off')

        if config.n_gpu > 1:
            raise NotImplementedError('Multi-GPU is not supported yet')

        if is_master():
            self.checkpt_dir.mkdir()
            # setup visualization writer instance
            log_dir.mkdir()
            self.writer = TensorboardWriter(
                log_dir, cfg_trainer['tensorboard'])
        else:
            self.writer = TensorboardWriter(log_dir, False)

        self.criterion_a = criterion_a
        self.optimizer_a = optimizer_a
        self.lr_scheduler_a = lr_scheduler_a

        self.criterion_b = criterion_b
        self.optimizer_b = optimizer_b
        self.lr_scheduler_b = lr_scheduler_b

        self.metric_ftns_a = metrics_a
        self.metric_ftns_b = metrics_b
        metric_names_a = ['loss'] + \
            [met.__name__ for met in self.metric_ftns_a]
        metric_names_b = ['loss'] + \
            [met.__name__ for met in self.metric_ftns_b]
        self.ep_metrics_a = EpochMetrics(metric_names_a, phases=(
            'train_a', 'valid_a'), monitoring=self.monitor, writer=self.writer)
        self.ep_metrics_b = EpochMetrics(metric_names_b, phases=(
            'train_b', 'valid_b'), monitoring=self.monitor, writer=self.writer)

        args_a = ['loss', *[m.__name__ for m in self.metric_ftns_a]]
        args_b = ['loss', *[m.__name__ for m in self.metric_ftns_b]]

        self.train_metrics_a = BatchMetrics(
            *args_a, postfix='/train_a', writer=self.writer)
        self.valid_metrics_a = BatchMetrics(
            *args_a, postfix='/valid_a', writer=self.writer)
        self.train_metrics_b = BatchMetrics(
            *args_b, postfix='/train_b', writer=self.writer)
        self.valid_metrics_b = BatchMetrics(
            *args_b, postfix='/valid_b', writer=self.writer)

        self.saving_latest_k = cfg_trainer.get('saving_latest_k', -1)

        write_conf(self.config, 'config.yaml')

        self.start_epoch_a = 1
        self.start_epoch_b = 1

        self.resume_flag = False
        self.resume_conf = config.get(
            'resume_conf', [])  # resume epoch index
        if config.resume_a is not None:
            self._resume_checkpoint('a', self.resume_conf)
            self.resume_flag = True
        if config.resume_b is not None:
            self._resume_checkpoint('b', self.resume_conf)
            self.resume_flag = True

        # dataloader init
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.limit_train_iters = config['trainer'].get(
            'limit_train_iters', len(self.data_loader))
        if not self.limit_train_iters or self.limit_train_iters > len(self.data_loader):
            self.limit_train_iters = len(self.data_loader)
        self.limit_valid_iters = config['trainer'].get(
            'limit_valid_iters', len(self.valid_data_loader))
        if not self.limit_valid_iters or self.limit_valid_iters > len(self.valid_data_loader):
            self.limit_valid_iters = len(self.valid_data_loader)

    def train(self):
        """
        Full training logic
        """

        # get starting point of training workflow
        if 'epoch' in self.resume_conf and self.resume_flag:
            start_point = self._get_start_point()
            w0, n0, a0, b0 = start_point['work_idx'], start_point['n_idx'], start_point['a_idx'], start_point['b_idx']
        else:
            w0, n0, a0, b0 = 0, 0, 1, 1

        ep_cnt_a = self.start_epoch_a
        ep_cnt_b = self.start_epoch_b

        # training
        for work_idx in range(w0, len(self.workflow)):
            work = self.workflow[work_idx]
            for n_idx in range(n0, work['n']):
                if work['mode'] == 'alter':
                    # alternative training of net_a & net_b
                    for op_i in work['op']:
                        if op_i[0] == 'a':
                            for _ in range(a0, op_i[1]+1):
                                time_start = time.time()
                                result = self._train_epoch_a(epoch=ep_cnt_a)
                                time_end = time.time()
                                self.ep_metrics_a.update(ep_cnt_a, result)
                                self._after_epoch(
                                    'a', ep_cnt_a, result, time_end-time_start)
                                ep_cnt_a += 1
                            a0 = 1  # reset counter
                        elif op_i[0] == 'b':
                            for _ in range(b0, op_i[1]+1):
                                time_start = time.time()
                                result = self._train_epoch_b(epoch=ep_cnt_b)
                                time_end = time.time()
                                self._after_epoch(
                                    'b', ep_cnt_b, result, time_end-time_start)
                                ep_cnt_b += 1
                            b0 = 1  # reset counter

                elif work['mode'] == 'e2e':
                    # end-to-end training of net_a & net_b
                    time_start = time.time()
                    result_a, result_b = self._train_epoch_ab(
                        ep_cnt_a, ep_cnt_b, n_idx=n_idx, n=work['n'])
                    time_end = time.time()
                    self._after_epoch('a', ep_cnt_a, result_a,
                                      time_end-time_start)
                    self._after_epoch('b', ep_cnt_b, result_b,
                                      time_end-time_start)

                    ep_cnt_a += 1
                    ep_cnt_b += 1

                else:
                    raise NotImplementedError(
                        f"training mode ({work[0]}) should be 'e2e' | 'alter' ")
                n0 = 0  # restart counter

    def _get_start_point(self):
        # get the start point of training workflow
        cnt = {'a': 0, 'b': 0}
        for work_idx, work in enumerate(self.workflow):
            for n_idx in range(work['n']):
                cnt_a0, cnt_b0 = cnt['a'], cnt['b']  # current cnt
                if work['mode'] == 'alter':
                    for op_i in work['op']:
                        cnt[op_i[0]] += op_i[1]
                elif work['mode'] == 'e2e':
                    cnt['a'] += 1
                    cnt['b'] += 1
                else:
                    raise NotImplementedError(
                        f"training mode ({work[0]}) should be 'e2e' | 'alter' ")

                if (cnt['a'] >= self.start_epoch_a or cnt['a'] == 0) and (cnt['b'] >= self.start_epoch_b or cnt['b'] == 0):
                    a_idx = self.start_epoch_a - cnt_a0
                    b_idx = self.start_epoch_b - cnt_b0
                    assert a_idx > 0 and b_idx > 0, f'start_epoch_a ({self.start_epoch_a}) and start_epoch_b ({self.start_epoch_b}) are unpaired start-points for workflow {self.workflow}. Considering not to resume "epoch" in $resume_conf to avoid this error'
                    return {'work': work, 'work_idx': work_idx, 'n_idx': n_idx, 'a_idx': a_idx, 'b_idx': b_idx}
        raise ValueError(
            f'start_epoch_a({self.start_epoch_a}) and start_epoch_b({self.start_epoch_b}) are unpaired start-points for workflow {self.workflow}. Considering not to resume "epoch" in $resume_conf to avoid this error')

    def _after_epoch(self, model_id, ep, result, time_cost):
        # choose model
        ep_metrics_name = 'ep_metrics_' + model_id
        ep_metrics = getattr(self, ep_metrics_name)

        # write result metrics to tensorboard
        self.writer.set_step(ep)
        ep_metrics.update(ep, result)

        # print result metrics of this epoch
        max_line_width = max(len(line)
                             for line in str(ep_metrics).splitlines())
        # divider ---
        self.logger.info('-' * max_line_width)
        self.logger.info('\n' + str(ep_metrics.latest()) + '\n')

        ep_metrics.to_csv(f'epoch-results-{model_id}.csv')

        # divider ===
        self.logger.info(
            f'Epoch Time Cost: {time_cost:.2f}s')
        self.logger.info('=' * max_line_width)

        # saving checkpoint
        self._save_checkpoint(
            ep, model_id, keep_latest_k=self.saving_latest_k)

    def _after_iter(self, epoch, batch_idx, phase, model_id, loss, iter_metrics, image_tensors: dict):
        self.writer.set_step(
            (epoch - 1) * getattr(self, f'limit_{phase}_iters') + batch_idx, speed_chk=f'{phase}_{model_id}')

        loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
        getattr(self, f'{phase}_metrics_{model_id}').update('loss', loss_v)

        for k, v in iter_metrics.items():
            getattr(self, f'{phase}_metrics_{model_id}').update(k, v)

        for k, v in image_tensors.items():
            self.writer.add_image(
                f'{phase}_{model_id}/{k}', make_grid(image_tensors[k].cpu(), nrow=2, normalize=True))

    def _save_checkpoint(self, epoch, model_id, keep_latest_k=-1, save_latest=True):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param model_id: which model to save, a | b
        :param keep_latest_k: keep latest k ckp and remove the old ones. Default=-1, keep all ckp
        :param save_latest: if True, save a copy of current checkpoint file as 'model_latest.pth'
        """
        # choose model
        model = getattr(self, 'model_'+model_id)
        optimizer = getattr(self, 'optimizer_' + model_id)

        # make data
        arch = type(model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': self.config
        }

        # save new ckp
        filename = str(self.checkpt_dir /
                       f'checkpoint-{model_id}-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(
            f"Model checkpoint saved at: \n    {os.getcwd()}/{filename}")

        # remove old ckp
        if keep_latest_k > 1 and epoch > keep_latest_k:
            old_filename = str(self.checkpt_dir /
                               f'checkpoint-{model_id}-epoch{epoch-keep_latest_k}.pth')
            try:
                os.remove(old_filename)
            except FileNotFoundError:
                # this happens when current model is loaded from checkpoint
                # or target file is already removed somehow
                pass
        latest_path = str(self.checkpt_dir / f'model-{model_id}-latest.pth')
        copyfile(filename, latest_path)

    def _resume_checkpoint(self, model_id, resume_conf=['epoch', 'optimizer']):
        """
        Resume from saved checkpoints

        :param resume_conf: resume config that controls what to resume
        """

        resume_path = opj(os.getcwd(), self.config['resume_'+model_id])

        self.logger.info(
            f"Loading checkpoint for model_{model_id}: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.logger.info(
            f"model_{model_id} checkpoint (epoch {checkpoint['epoch']}) loaded!")

        # load architecture params from checkpoint.
        if checkpoint['config'].get('arch_'+model_id, None) != self.config.get('arch_'+model_id, None):
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        getattr(self, f"model_{model_id}").load_state_dict(
            checkpoint['state_dict'])

        # load optimizer state from checkpoint
        if 'optimizer' in resume_conf:
            getattr(self, f"optimizer_{model_id}").load_state_dict(
                checkpoint['optimizer'])
            self.logger.info(
                f'optimizer_{model_id} resumed from the loaded checkpoint!')

        # epoch start point
        if 'epoch' in resume_conf:
            setattr(self, 'start_epoch_'+model_id, checkpoint['epoch'] + 1)
            self.logger.info(
                f"Start training model_{model_id} from resumed epoch ({checkpoint['epoch']}).")
        else:
            setattr(self, 'start_epoch_'+model_id, 1)
            self.logger.info(
                f"Start training model_{model_id} from restarted epoch (1).")

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        try:
            # epoch-based training
            # total = len(self.data_loader.dataset)
            total = self.data_loader.batch_size * self.limit_train_iters
            current = batch_idx * self.data_loader.batch_size
            if dist.is_initialized():
                current *= dist.get_world_size()
        except AttributeError:
            # iteration-based training
            total = self.limit_train_iters
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)

    def _train_epoch_a(self, epoch, *args, **kwargs):
        raise NotImplementedError

    def _train_epoch_b(self, epoch, *args, **kwargs):
        raise NotImplementedError

    def _train_epoch_ab(self, epoch_a, epoch_b, *args, **kwargs):
        raise NotImplementedError

    def _valid_epoch_a(self, epoch, *args, **kwargs):
        raise NotImplementedError

    def _valid_epoch_b(self, epoch, *args, **kwargs):
        raise NotImplementedError

    def _valid_epoch_ab(self, epoch_a, epoch_b, *args, **kwargs):
        raise NotImplementedError
