import argparse
import os

import addict
import fire
import torch
import pandas as pd
from torch.backends import cudnn
import getters
import training
from training.config import parse_config, save_config
from training.runner import GPUNormRunner
import torch.nn as nn
import torch.nn.functional as F
cudnn.benchmark = True
import torch.distributed as dist
from torch.nn import DataParallel
import random
import numpy as np
import time


def worker_init_fn(seed):

    seed = (seed + 1)
    np.random.seed(seed)
    random.seed(seed)
    random.Random().seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SegmentationScale(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        scale: float
    ):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x):
        oldsize = x.shape[-1]
        x = F.interpolate(x, scale_factor=self.scale)
        x = self.model(x)

        if self.training:
            return x
        x = F.interpolate(x, size=[oldsize, oldsize], mode="bilinear")
        return x

def kep_path(keppath):
    path = os.path.dirname(keppath)
    kepname = os.path.basename(keppath)
    file_list = os.listdir(path)

    for file_name in file_list:
        if kepname in file_name:
            return os.path.join(path, file_name)
    return ""

def main(cfg):

    # set GPUS
    if cfg.distributed:

        torch.distributed.init_process_group(backend="nccl")
        cfg.lrank = torch.distributed.get_rank()
        # setup_seed(0 + cfg.lrank)
        print("--------------", cfg.lrank)
        torch.cuda.set_device(cfg.lrank)

    else:
        setup_seed(0)
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.gpus)) if cfg.get("gpus") else ""


    # --------------------------------------------------
    # define model
    # --------------------------------------------------
    model_recover = False
    init_epoch = 0
    model_output_keys = cfg.model.model_output_keys


    print('Creating model...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = getters.get_model(architecture=cfg.model.architecture, init_params=cfg.model.init_params)

    cfg.model.preweightpath = kep_path(cfg.model.preweightpath)

    if os.path.isfile(cfg.model.preweightpath):
        print("model load from -> ", cfg.model.preweightpath)
        state_dict = torch.load(cfg.model.preweightpath, map_location=torch.device('cpu'))["state_dict"]
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    optimizer_dict = None
    if os.path.exists(os.path.join(cfg.logdir, 'checkpoints', 'last.pth')):
        # 自动恢复训练
        state_all = torch.load(os.path.join(cfg.logdir, 'checkpoints', 'last.pth'), map_location=torch.device('cpu'))
        state_dict = state_all["state_dict"]
        optimizer_dict = state_all["optimizer"]
        init_epoch = state_all["epoch"]+1
        print("Recover train, from epoch ", str(init_epoch))
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})


    if cfg.model.model_sacle != 1:
        print('moedl sacel is ', cfg.model.model_sacle)
        model = SegmentationScale(model, scale=cfg.model.model_sacle)

    print('Moving model to device...')

    model.to(device)
    if cfg.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print('Collecting model parameters...')
    params = model.parameters()

    # --------------------------------------------------
    # define datasets and dataloaders
    # --------------------------------------------------
    print('Creating datasets and loaders..')
    df = pd.read_csv(cfg.data.df_path, dtype={'id': object})
    train_ids = df[df['fold'] != int(cfg.data.fold)]['id'].tolist()
    valid_ids = df[df['fold'] == int(cfg.data.fold)]['id'].tolist()

    assert (len(train_ids)) != 0
    assert not set(train_ids).intersection(set(valid_ids))

    train_dataset = getters.get_dataset(
        name=cfg.data.train_dataset.name,
        ids=train_ids,
        init_params=cfg.data.train_dataset.init_params,
    )

    train_sampler = None
    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **cfg.data.train_dataloader, sampler=train_sampler, shuffle=train_sampler is None,
        worker_init_fn=worker_init_fn
    )

    if 'valid_dataset' in cfg.data:
        valid_dataset = getters.get_dataset(
            name=cfg.data.valid_dataset.name,
            ids=valid_ids,
            init_params=cfg.data.valid_dataset.init_params,
        )

        val_sampler = None
        if cfg.val_distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, **cfg.data.valid_dataloader, sampler=val_sampler,
        )
    else:
        valid_dataloader = None

    # --------------------------------------------------
    # define losses and metrics functions
    # --------------------------------------------------
    losses = {}
    for output_name in cfg.training.losses.keys():
        loss_name = cfg.training.losses[output_name].name
        loss_init_params = cfg.training.losses[output_name].init_params
        losses[output_name] = getters.get_loss(loss_name, loss_init_params).to(device)


    metrics = {}
    for output_name in cfg.training.metrics.keys():
        metrics[output_name] = []
        for metric in cfg.training.metrics[output_name]:
            metrics[output_name].append(
                getters.get_metric(metric.name, metric.init_params)
            )

    # --------------------------------------------------
    # define optimizer and scheduler
    # --------------------------------------------------
    print('Defining optimizers and schedulers..')

    optimizer = getters.get_optimizer(
        cfg.training.optimizer.name,
        model_params=params,
        init_params=cfg.training.optimizer.init_params,
    )

    if os.path.exists(os.path.join(cfg.logdir, 'checkpoints', 'last.pth')):
        optimizer.load_state_dict({k.replace('module.', ''): v for k, v in optimizer_dict.items()})

    if cfg.training.get("scheduler", None):
        scheduler = getters.get_scheduler(
            cfg.training.scheduler.name,
            optimizer,
            cfg.training.scheduler.init_params,
        )
    else:
        scheduler = None

    # --------------------------------------------------
    # define callbacks
    # --------------------------------------------------
    print('Defining callbacks..')
    callbacks = []

    # add scheduler callback
    listen_key = None

    if scheduler is not None:
        if cfg.training.scheduler.monitor != None:
            if cfg.training.scheduler.name == 'ReduceLROnPlateau':
                listen_key = cfg.training.scheduler.monitor
            callbacks.append(training.callbacks.Scheduler(
                scheduler,
                sc_name=cfg.training.scheduler.name,
                monitor=cfg.training.scheduler.monitor))

    # add default logging and checkpoint callbacks
    if cfg.logdir is not None:
        # tb logging
        callbacks.append(training.callbacks.TensorBoard(
            os.path.join(cfg.logdir, 'tb')
        ))
        st = 10
        if 'save_top' in cfg:
            st = cfg.save_top
        # checkpointing
        callbacks.append(training.callbacks.ModelCheckpoint(
            directory=os.path.join(cfg.logdir, 'checkpoints'),
            monitor=cfg.training.scheduler.monitor,
            save_best=True,
            save_last=True,
            save_top_k=st,
            mode="max",
            verbose=True,
        ))

    # --------------------------------------------------
    # model pallel and mix train
    # --------------------------------------------------

    if cfg.distributed:
        print("Creating distributed Model on gpus:", cfg.lrank)
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[cfg.lrank])
    else:
        print("Creating DataParallel Model on gpus:", cfg.gpus)
        model = DataParallel(model).to(device)
    # --------------------------------------------------
    # start training
    # --------------------------------------------------
    print('Start training...')
    runner = GPUNormRunner(model,
                           model_output_keys=model_output_keys,
                           model_device=device,
                           local_rank=cfg.lrank,
                           fp16=cfg.fp16,
                           is_distribe=cfg.distributed,
                           is_val_distribe=cfg.val_distributed,
                           train_sampler=train_sampler)
    runner.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
    )

    runner.fit(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        callbacks=callbacks,
        logdir=cfg.logdir,
        listen_key=listen_key,
        initial_epoch=init_epoch,
        **cfg.training.fit,
    )


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    random.Random().seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    setup_seed(0)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default="config/Final_11_8/effb3_change_cad_102_s15_FDA_alltrain.yaml",
                        type=str, dest='configs', help='The file of the hyper parameters.')
    parser.add_argument('--flod_csv', default="config/HUWEI_5folds.csv",
                        type=str, dest='flod_csv', help='The file of the hyper parameters.')
    parser.add_argument('--data_path', default="D:/Dataset/ChangeDetection/Huawei_2021/train",
                        type=str, dest='data_path', help='The file of the hyper parameters.')
    parser.add_argument('--local_rank', default=0,
                        type=int, dest='local_rank', help='local rank of current process')

    args_parser = parser.parse_args()

    args_parser.configs = args_parser.configs.replace('\r', '')
    cfg = addict.Dict(parse_config(config=args_parser.configs))
    logdir = cfg.get("logdir", None)

    cfg.data.df_path = args_parser.flod_csv
    cfg.data.train_dataset.init_params.root_dir = args_parser.data_path
    if 'valid_dataset' in cfg.data:
        cfg.data.valid_dataset.init_params.root_dir = args_parser.data_path

    if logdir is not None:
        save_config(cfg.to_dict(), logdir, name="config.yml")
        print(f"Config saved to: {logdir}")

    print("----", cfg.distributed)
    main(cfg)
    os._exit(0)

