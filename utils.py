"""
General helpers: logging, seeding, GPU setup, config parsing and a simple
LR-warmup wrapper.
"""
import os
import torch
import random
import logging
import argparse

import numpy as np
from mmengine.config import Config, DictAction


def get_logger(filepath, log_info):
    """Create a file logger. `filepath` is the log file, `log_info` is a
    header string written once at creation time."""
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger


def init_seed(args):
    """Seed torch/numpy/random and force deterministic cuDNN for reproducibility."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_gpu(cfg):
    """Restrict visible GPUs to `cfg.multi_gpu` (comma separated device ids,
    e.g. "0,1") and return whether more than one device was requested."""
    gpus = str(cfg.multi_gpu)  # may arrive as int from --cfg-options
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    multi_gpu = len(gpus.split(",")) > 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return multi_gpu, device


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear LR warmup for `warmup_steps` steps, then delegates to
    `base_scheduler`. Not used by default in train.py, kept as an option."""

    def __init__(self, optimizer, warmup_steps, base_scheduler, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_steps:
            super().step(epoch)
        else:
            self.base_scheduler.step(epoch)


def parse_args():
    """Parse CLI args and load the config file (mmengine `Config`).

    By default all hyperparameters live in the `--config` file. The extra
    CLI flags below only take effect when the config sets `from_command =
    True`, in which case they overwrite the corresponding config fields
    (handy for sweeping seed/loss/etc. from a shell script).
    """
    parser = argparse.ArgumentParser(description='Training configuration parser')
    parser.add_argument(
        '--config',
        help='Path to the training config file',
        default='configs/egoexo4d.py')
    parser.add_argument(
        '--seed',
        help='Random seed for reproducibility',
        type=int,
        default=8898)
    parser.add_argument(
        '--num_layers_ego',
        help='Number of decoder layers for the egocentric branch',
        type=int,
        default=3)
    parser.add_argument(
        '--num_layers_exo',
        help='Number of decoder layers for the exocentric branch',
        type=int,
        default=1)
    parser.add_argument(
        '--VIBGB',
        help='Enable the Variational-Information-Bottleneck-based Gradient Blending loss',
        action='store_true')
    parser.add_argument(
        '--CVML',
        help='Enable the InfoNCE (ego<->exo) contrastive loss',
        action='store_true')
    parser.add_argument(
        '--VIB',
        help='Enable the VIB (KL) regularization term',
        action='store_true')
    parser.add_argument(
        '--loss',
        help='Classification loss type: "ordinal" (CORN) or "cls" (cross entropy)',
        type=str,
        default="ordinal")
    parser.add_argument(
        '--vib_rate',
        help='Weight of the VIB KL loss term',
        type=float,
        default=0.02)
    parser.add_argument(
        '--cvml_rate',
        help='Weight of the InfoNCE loss term',
        type=float,
        default=0.02)
    parser.add_argument(
        '--bs_train',
        help='Training batch size',
        type=int,
        default=128)
    parser.add_argument(
        '--tau',
        help='Gumbel-Softmax temperature used for view/ego-exo gating',
        type=float,
        default=0.7)
    parser.add_argument(
        '--pretrain',
        help='Backbone feature set to load: "k400" | "k600" | "howto100"',
        type=str,
        default="k600")
    parser.add_argument(
        '--logname',
        help='Name used for the log file under exp/<logname>.log',
        type=str,
        default="2")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config fields, e.g. --cfg-options seed=1 gb_mode=fixed')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    if cfg.from_command:
        cfg.seed = args.seed
        cfg.num_layers_ego = args.num_layers_ego
        cfg.num_layers_exo = args.num_layers_exo
        cfg.VIBGB = args.VIBGB
        cfg.VIB = args.VIB
        cfg.loss = args.loss
        cfg.CVML = args.CVML
        cfg.vib_rate = args.vib_rate
        cfg.cvml_rate = args.cvml_rate
        cfg.bs_train = args.bs_train
        cfg.tau = args.tau
        cfg.pretrain = args.pretrain
        cfg.logname = args.logname
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    return cfg
