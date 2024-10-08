# ======================================
# BaseTrainer for basic network
# ======================================
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from abc import abstractmethod, ABCMeta
from torchvision.utils import make_grid
from pathlib import Path
from shutil import copyfile
from numpy import inf
import time
import os
from datetime import datetime
from srcs.utils._util import write_conf, is_master, get_logger, collect
from srcs.logger import TensorboardWriter, EpochMetrics, BatchMetrics
from os.path import join as opj


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for all trainers
    """

    def __init__(self, model, config, criterion, metrics, optimizer, lr_scheduler=None, train_data_loader=None, valid_data_loader=None, test_data_loader=None):
        ## param assignment
        self.config = config
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        cfg_trainer = config['trainer']

        ## model
        self.device = config.local_rank if config.n_gpus > 1 else 0
        self.model = model.to(self.device)
        if config.n_gpus > 1:
            # multi GPU
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = DistributedDataParallel(
                model, device_ids=[self.device], output_device=self.device)
        # resume from checkpoint
        if config.resume is not None: 
            resume_conf = config.get(
                'resume_conf', None)
            if resume_conf is None:
                resume_conf = ['epoch', 'optimizer']
            self._resume_checkpoint(config.resume, resume_conf)

        ## logger
        self.logger = get_logger('trainer')
        self.logging_step = cfg_trainer.get('logging_step', 100)
        self.log_weight = config['trainer'].get('log_weight', False)
        self.checkpt_dir = Path(self.config.checkpoint_dir)
        log_dir = Path(self.config.log_dir)
        if is_master():
            self.checkpt_dir.mkdir(exist_ok=True)
            log_dir.mkdir(exist_ok=True)
            self.writer = TensorboardWriter(
                log_dir, cfg_trainer['tensorboard'])
        else:
            self.writer = TensorboardWriter(log_dir, False)

        ## metrics
        args = ['loss', *[m.__name__ for m in self.metrics]]
        self.train_metrics = BatchMetrics(
            *args, postfix='/train', writer=self.writer)
        self.valid_metrics = BatchMetrics(
            *args, postfix='/valid', writer=self.writer)
        metric_names = ['loss'] + [met.__name__ for met in self.metrics]
        # metric monitor:  monitoring model performance and saving best-checkpoint
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.ep_metrics = EpochMetrics(metric_names, phases=(
            'train', 'valid'), monitoring=self.monitor)

 
        ## runtime
        self.start_epoch = 1
        self.epochs = cfg_trainer.get('epochs', int(1e10))
        if self.epochs is None:
            self.epochs = int(1e10)
        # limit train/valid iters
        self.limit_train_iters = config['trainer'].get(
            'limit_train_iters', len(self.data_loader))
        if not self.limit_train_iters or self.limit_train_iters > len(self.data_loader):
            self.limit_train_iters = len(self.data_loader)
        self.limit_valid_iters = config['trainer'].get(
            'limit_valid_iters', len(self.valid_data_loader))
        if not self.limit_valid_iters or self.limit_valid_iters > len(self.valid_data_loader):
            self.limit_valid_iters = len(self.valid_data_loader)
        self.saving_top_k = cfg_trainer.get('saving_top_k', -1)
        self.milestone_ckp = cfg_trainer.get('milestone_ckp', [])
        self.early_stop = cfg_trainer.get('early_stop', inf)
        if self.early_stop is None:
            self.early_stop = inf
        # final test
        self.final_test = cfg_trainer.get('final_test', False)
        if self.final_test:
            if test_data_loader is None:
                self.logger.warning(
                    "Warning: test dataloader for final test is None, final test is omitted")
                self.final_test = False
            else:
                self.test_data_loader = test_data_loader
                self.final_test_dir = Path(self.config.final_test_dir)
                if is_master():
                    self.final_test_dir.mkdir(exist_ok=True)
        
        ## save conf info
        write_conf(self.config, 'config.yaml')


    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _test_epoch(self):
        """
        Final test logic after the training (! test the latest checkpoint)

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        self.logger.info(
            f"\n⏩⏩ Start Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ⏩⏩\n")
        not_improved_count = 0
        train_start = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            # train one epoch
            epoch_start = time.time()
            result = self._train_epoch(epoch) 
            self.ep_metrics.update(epoch, result)
            epoch_end = time.time()

            # print result metrics of this epoch
            max_line_width = max(len(line)
                                 for line in str(self.ep_metrics).splitlines())
            # divider ---
            self.logger.info('-' * max_line_width)
            self.logger.info('\n' + str(self.ep_metrics.latest()) + '\n')

            if is_master():
                # check if model performance improved or not, for early stopping and topk saving
                is_best = False
                improved = self.ep_metrics.is_improved()
                if improved:
                    not_improved_count = 0
                    is_best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    if self.final_test:
                        self.logger.info(
                            f"\n🎉🎉 Finish Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 🎉🎉\n\n == = > Start Testing(Using Latest Checkpoint): \n")
                        self._test_epoch()
                    else:
                        self.logger.info(
                            f"\n🎉🎉 Finish Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}🎉🎉\n\n")
                    exit(1)

                using_topk_save = self.saving_top_k > 0
                self._save_checkpoint(
                    epoch, save_best=is_best, save_latest=using_topk_save, milestone_ckp=self.milestone_ckp)
                # keep top-k checkpoints only, using monitoring metrics
                if using_topk_save:
                    self.ep_metrics.keep_topk_checkpt(
                        self.checkpt_dir, self.saving_top_k)

                self.ep_metrics.to_csv('epoch-results.csv')

            
            self.logger.info(
                f'🕒 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Epoch Time Cost: {epoch_end-epoch_start:.2f}s, Total Time Cost: {(epoch_end-train_start)/3600:.2f}h\n')
            self.logger.info('=' * max_line_width)
            if self.config.n_gpus > 1:
                dist.barrier()
        if self.final_test:
            self.logger.info(
                f"\n🎉🎉 Finish Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 🎉🎉\n\n == = > Start Testing(Using Latest Checkpoint): \n")
            self._test_epoch()
        else:
            self.logger.info(
                f"\n🎉🎉 Finish Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}🎉🎉\n\n")

    def _after_iter(self, epoch, batch_idx, phase, loss, metrics, image_tensors: dict):
        # TBD
        # hook after iter
        self.writer.set_step(
            (epoch - 1) * getattr(self, f'limit_{phase}_iters') + batch_idx, speed_chk=f'{phase}')

        loss_v = loss.item() if self.config.n_gpus == 1 else collect(loss)
        getattr(self, f'{phase}_metrics').update('loss', loss_v)

        for k, v in metrics.items():
            getattr(self, f'{phase}_metrics').update(
                k, v.item())  # `v` is a torch tensor

        for k, v in image_tensors.items():
            self.writer.add_image(
                f'{phase}/{k}', make_grid(image_tensors[k].cpu(), nrow=2, normalize=True))

    def _after_epoch(self, ep, result, epoch_time, total_time):
        # TBD
        # hook after epoch
        # choose model
        ep_metrics = getattr(self, 'ep_metrics')

        # write result metrics to tensorboard
        self.writer.set_step(ep)
        ep_metrics.update(ep, result)

        # print result metrics of this epoch
        max_line_width = max(len(line)
                             for line in str(ep_metrics).splitlines())
        # divider ---
        self.logger.info('-' * max_line_width)
        self.logger.info('\n' + str(ep_metrics.latest()) + '\n')

        ep_metrics.to_csv(f'epoch-results.csv')

        # divider ===
        self.logger.info(
            f'🕒 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Epoch Time Cost: {epoch_time:.2f}s, Total Time Cost: {total_time/3600:.2f}h\n')
        self.logger.info('=' * max_line_width)

        # saving checkpoint
        # self._save_checkpoint(
        #     ep, save_best=is_best, save_latest=using_topk_save)

    def _save_checkpoint(self, epoch, save_best=False, save_latest=True, milestone_ckp=[]):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, save a copy of current checkpoint file as 'model_best.pth'
        :param save_latest: if True, save a copy of current checkpoint file as 'model_latest.pth'
        :param milestone_ckp: save and keep current checkpoints if current epoch is in this milestone_ckp
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'epoch_metrics': self.ep_metrics, # may cause can't pickle error in torch.save
            'config': self.config
        }

        filename = str(self.checkpt_dir / f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(
            f"💾 Model checkpoint saved at:\n\t{filename}")  # {os.getcwd()}/
        if save_latest:
            latest_path = str(self.checkpt_dir / 'model_latest.pth')
            copyfile(filename, latest_path)
        if save_best:
            best_path = str(self.checkpt_dir / 'model_best.pth')
            copyfile(filename, best_path)
            self.logger.info(
                f"🔄 Renewing best checkpoint!")
        if milestone_ckp and epoch in milestone_ckp:
            landmark_path = str(
                self.checkpt_dir / f'model_epoch{epoch}.pth')
            copyfile(filename, landmark_path)
            self.logger.info(
                f"🔖 Saving milestone checkpoint at epoch {epoch}!")

    def _resume_checkpoint(self, resume_path, resume_conf=['epoch', 'optimizer']):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        :param resume_conf: resume config that controls what to resume
        """

        resume_path = opj(os.getcwd(), self.config['resume'])
        self.logger.info(f"📥 Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        if checkpoint['config'].get('arch', None) != self.config.get('arch', None):
            self.logger.warning("⚠️ Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")

        # preprocess DDP saved cpk
        if checkpoint['config'].get('arch', 1) > 1:
            state_dict = {k.replace('module.', ''): v for k,
                          v in state_dict.items()}

        # load cpk
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint
        if 'optimizer' in resume_conf:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(
                f'📣 Optimizer resumed from the loaded checkpoint!')

        # epoch start point
        if 'epoch' in resume_conf:
            self.start_epoch = checkpoint['epoch'] + 1
            self.logger.info(
                f"📣 Epoch index resumed to epoch ({checkpoint['epoch']}).")
        else:
            self.start_epoch = 1
            self.logger.info(
                f"📣 Epoch index renumbered from epoch (1).")

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
