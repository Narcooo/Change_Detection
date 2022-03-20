import os
import re
import sys
import cv2
import time
import torch
import csv
from typing import Optional, Type
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Mapping, Union
from collections import defaultdict
import numpy as np
from .SCDD_eval import Evalmetric
from .callbacks import CallbackList


def graytorgb(img):
    label_mapping_rgb = {0:(255, 255, 255),
                         1:(0, 0, 255),
                         2:(128, 128, 128),
                         3:(0, 128, 0),
                         4:(0, 255, 0),
                         5:(128, 0, 0),
                         6:(255, 0, 0),}
    img_rgb = np.stack(np.vectorize(label_mapping_rgb.get)(img),  axis=2).astype('uint8')
    return img_rgb

def to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_name(instance):
    if hasattr(instance, '__name__'):
        return instance.__name__
    else:
        return to_snake(instance.__class__.__name__)


def timeit(f):
    def wrapped(*args, **kwargs):
        # start = time.time()
        res = f(*args, **kwargs)
        # print(f"{f.__name__}: {time.time() - start}")
        return res

    return wrapped


class Meter:

    def __init__(self):
        self._data = defaultdict(list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k.find('micro') != -1:
                self._data[k] = [v.item()]
            else:
                self._data[k].append(v.item())

    def data(self, prefix=None):
        prefix = '{}_'.format(prefix) if prefix is not None else ''
        return {(prefix + name): values for name, values in self._data.items()}

    def mean(self, prefix=None):
        prefix = '{}_'.format(prefix) if prefix is not None else ''
        return {(prefix + name): sum(values) / len(values) for name, values in self._data.items()}

    def last(self, prefix=None):
        prefix = '{}_'.format(prefix) if prefix is not None else ''
        return {(prefix + name): values[-1] for name, values in self._data.items()}


class Runner:

    def __init__(
            self,
            model: torch.nn.Module,
            model_device: Union[str, torch.device],
            model_input_keys: Optional[Union[str, List[str]]] = 'image',
            model_output_keys: Optional[Union[str, List[str]]] = 'mask',
            is_distribe=False,
            is_val_distribe=False,
            local_rank=None,
            fp16=False,
            train_sampler=None
    ):

        self.model = model
        self.device = model_device
        self.is_val_distribe = is_val_distribe
        self.input_keys = model_input_keys if isinstance(model_input_keys, (list, tuple)) else [model_input_keys]
        self.output_keys = model_output_keys if isinstance(model_output_keys, (list, tuple)) else [
            model_output_keys]

        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.local_rank = local_rank
        self.fp16 = fp16
        self._to_device(self.model)
        self.train_sampler = train_sampler
        self.is_distribe = is_distribe

    def compile(
            self,
            optimizer: Optional[Type[torch.optim.Optimizer]] = None,
            loss: Mapping[str, callable] = None,
            metrics: Mapping[str, List[callable]] = None
    ) -> None:
        self.optimizer = optimizer
        self.loss = self._to_device(loss)
        self.metrics = self._to_device(metrics)


    @timeit
    def _to_device(self, x):

        if isinstance(x, (list, tuple)):
            return [self._to_device(xi) for xi in x]
        elif isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        else:
            if hasattr(x, 'to'):
                return x.to(self.device)
            else:
                return x

    def _model_to_mode(self, mode='train'):
        if mode == 'train' and hasattr(self.model, 'train'):
            self.model.train()
        elif mode == 'eval' and hasattr(self.model, 'eval'):
            self.model.eval()
        else:
            warnings.warn(
                "Model does not support train/eval modes, are you using traced module?",
                UserWarning,
            )

    def _prepare_input(self, batch: Mapping[str, torch.Tensor]) -> List:
        """Collect model input data from batch (collect list)"""
        if not isinstance(batch, dict):
            raise ValueError("Runner expect batches to be of type Dict! Got type {}.".format(type(batch)))
        return [batch[k] for k in batch if k in self.input_keys]

    def _prepare_output(self, model_output: Union[torch.Tensor, list, tuple, dict]) -> Mapping[str, torch.Tensor]:
        """Take model output and convert it it dict, if it is not dict"""


        if isinstance(model_output, torch.Tensor):
            model_output = [model_output]

        if isinstance(model_output, (list, tuple)):
            if len(self.output_keys) == 1 :
                model_output = [model_output]
            elif len(model_output) != len(self.output_keys):
                raise ValueError("Runner have output keys {}, but model produce only {} outputs".format(
                    self.output_keys, len(model_output))
                )
            output = {k: v for k, v in zip(self.output_keys, model_output)}

        elif isinstance(model_output, dict):
            output = {k: model_output[k] for k in self.output_keys}
        else:
            raise ValueError("Model output expected to be list, dict or Tensor, got {}.".format(type(model_output)))
        return output

    @timeit
    def _feed_batch(self, batch) -> Mapping[str, torch.Tensor]:
        input = self._prepare_input(batch)
        output = self.model(*input)
        output = self._prepare_output(output)
        return output

    @timeit
    def _compute_losses(
            self,
            output: Mapping[str, torch.Tensor],
            target: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:

        losses_dict = {}

        # compute loss for each output
        for output_name, criterion in self.loss.items():
            loss_name = 'loss_{}'.format(output_name)
            losses_dict[loss_name] = criterion(output[output_name], target[output_name])

        # compute total loss across all outputs
        losses_dict['loss'] = sum(loss for loss in losses_dict.values())

        return losses_dict

    @timeit
    def _compute_metrics(
            self,
            output: Mapping[str, torch.Tensor],
            target: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        metrics_dict = {}
        for output_name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                metric_name = '{output_name}_{metric_name}'.format(
                    output_name=output_name,
                    metric_name=get_name(metric),
                )
                metric_value = metric(
                    output[output_name],
                    target[output_name],
                )
                if metric_name in metrics_dict.keys():
                    metric_name = metric_name + '_' + str(i)
                metrics_dict[metric_name] = metric_value
        return metrics_dict

    def _reset_metrics(self, is_distribe = False):
        for output_name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                if hasattr(metric, "reset"):
                    metric.reset(is_distribe)

    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> None:
        total_loss = loss / accumulation_steps
        if self.fp16:
            with amp.scale_loss(loss / accumulation_steps, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

    def _update_weights(self):
        self.optimizer.step()
        torch.cuda.synchronize()
        self.model.zero_grad()

    def _format_logs(self, logs):
        str_values = ['{}:{:.4f}'.format(k, v) for k, v in logs.items()]
        str_log = ', '.join(str_values)
        return str_log

    def fit(
            self,
            train_dataloader,
            train_steps=None,
            valid_dataloader=None,
            valid_steps=None,
            accumulation_steps=1,
            verbose=True,
            logdir=None,
            epochs=1,
            initial_epoch=0,
            callbacks=None,
            listen_key=None,
    ) -> dict:

        if self.loss is None:
            raise ValueError('Provide loss for training!')

        # define training callbacks
        logs = {}
        callbacks = CallbackList(callbacks or [])
        callbacks.set_runner(self)
        callbacks.on_train_begin(logs=logs)

        # start training loop
        for epoch in range(initial_epoch, epochs):
            if self.train_sampler is not None:  # 必须设置随机数种子，不然不会随机
                self.train_sampler.set_epoch(epoch)
            print('Epoch {}/{}'.format(epoch, epochs - 1))

            meter = Meter()
            self._reset_metrics()
            self._model_to_mode('train')
            callbacks.on_epoch_begin(epoch)

            if self.local_rank == 0:
                pbar = tqdm(total=train_steps or len(train_dataloader), desc='train', disable=not verbose, ncols=0)

            for i, batch in enumerate(train_dataloader):
                # batch begin callbacks
                callbacks.on_batch_begin(i)
                # main training process
                batch = self._to_device(batch)
                output = self._feed_batch(batch)
                losses = self._compute_losses(output, batch)
                self._backward(losses['loss'], accumulation_steps)
                if (i + 1) % accumulation_steps == 0:
                    self._update_weights()

                # collecting metrics
                metrics = dict()
                if self.metrics is not None:
                    metrics = self._compute_metrics(output, batch)

                # update batch logs
                meter.update(**losses, **metrics)
                batch_logs = meter.last()
                callbacks.on_batch_end(i, batch_logs)

                if verbose and self.local_rank == 0:
                    _logs_dict = meter.mean()
                    _logs_str = self._format_logs(_logs_dict)
                    pbar.set_postfix_str(_logs_str)
                    pbar.update()

                if train_steps is not None and i >= train_steps - 1:
                    break

            epoch_logs = meter.mean()

            # evaluation stage
            if valid_dataloader is not None:
                if self.is_val_distribe:
                    elog = self.evaluate(
                        valid_dataloader,
                        steps=valid_steps,
                        verbose=verbose,
                        logdir=logdir,
                    )
                elif self.local_rank == 0:
                    elog = self.evaluate(
                        valid_dataloader,
                        steps=valid_steps,
                        verbose=verbose,
                        logdir=logdir,
                    )

                if self.local_rank == 0:
                    epoch_logs.update(elog)

            if self.local_rank == 0:
                logs[epoch] = epoch_logs
                callbacks.on_epoch_end(epoch, epoch_logs)
                self.writercsv(logdir, epoch_logs)

        callbacks.on_train_end(logs)

        return logs

    @torch.no_grad()
    def evaluate(
            self,
            dataloader,
            steps=None,
            verbose=True,
            reduce=True,
            logdir=None,
    ):
        savepre_path = os.path.join(logdir, 'pre')
        # savedistance_path = os.path.join(logdir, 'pre_distance')
        savepre_path_before = os.path.join(logdir, 'pre_before')
        savepre_path_after = os.path.join(logdir, 'pre_after')

        if self.local_rank == 0:
            if os.path.exists(savepre_path) is False:
                os.makedirs(savepre_path)

            # if os.path.exists(savedistance_path) is False:
            #     os.makedirs(savedistance_path)

            if os.path.exists(savepre_path_before) is False:
                os.makedirs(savepre_path_before)

            if os.path.exists(savepre_path_after) is False:
                os.makedirs(savepre_path_after)

        if self.is_distribe:
            torch.distributed.barrier()

        if self.loss is None and self.metrics is None:
            raise ValueError('Provide metrics or/and losses for evaluation!')

        meter = Meter()
        self._reset_metrics(self.is_val_distribe)
        self._model_to_mode('eval')

        if self.local_rank == 0:
            pbar = tqdm(total=steps or len(dataloader), desc='valid', disable=not verbose, ncols=0)

        for i, batch in enumerate(dataloader):
            batch = self._to_device(batch)
            output = self._feed_batch(batch)
            losses = self._compute_losses(output, batch) if self.loss is not None else {}
            metrics = self._compute_metrics(output, batch) if self.metrics is not None else {}

            meter.update(**losses, **metrics)

            if verbose:
                _logs_dict = meter.last(prefix='val')
                _logs_str = self._format_logs(_logs_dict)
                if self.local_rank == 0:
                    if verbose != 2:
                        pbar.set_postfix_str(_logs_str)
                    pbar.update()

            if steps is not None and i >= steps - 1:
                break

        logs = meter.last(prefix='val') if reduce else meter.data(prefix='val')
        # val_change_mask_micro_iou, val_mask1_micro_iou, val_mask2_micro_iou
        #logs['seg_change'] = logs['val_change_mask_micro_f1'] * 0.6 + (
        #            logs['val_mask1_micro_f1'] + logs['val_mask2_micro_f1']) * 0.2
        if self.is_val_distribe:
            torch.distributed.barrier()
        return logs

    @torch.no_grad()
    def predict(
            self,
            dataloader,
            verbose=True,
            ignore_outputs=None,
    ):
        self._model_to_mode('eval')

        ignore_outputs = ignore_outputs or []
        output = {}

        with tqdm(dataloader, desc='infer',
                  disable=not verbose, file=sys.stdout) as p_dataloader:
            for i, batch in enumerate(p_dataloader):
                batch = self._to_device(batch)
                output = self._feed_batch(batch)

                for k in output.keys():
                    if k not in ignore_outputs:
                        output[k].append(
                            output[k].cpu().detach()
                        )

        output = {k: torch.cat(v, dim=0) for k, v in output.items()}
        return output

    @torch.no_grad()
    def predict_on_batch(self, batch):
        batch = self._to_device(batch)
        output = self._feed_batch(batch)
        return output

    def writercsv(self, logdir, data):
        logpath = os.path.join(logdir, 'log.csv')
        firstline = []
        logline = []
        for k, v in data.items():
            if isinstance(v, float):
                v = "%.6f" % (v)
                firstline.append(k)
                logline.append(v)

        if os.path.isfile(logpath):
            with open(logpath, 'a', newline='')as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(logline)
        else:
            with open(logpath, 'w', newline='')as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(firstline)
                csv_write.writerow(logline)


class ApexRunner(Runner):
    """Model and optimizer should be initialized with apex.amp.initialize(...)"""

    def _backward(self, loss: torch.Tensor, accumulation_steps: int = 1) -> None:
        if apex is not None:
            with apex.amp.scale_loss(loss / accumulation_steps, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            super()._backward(loss, accumulation_steps)


class GPUNormRunner(Runner):

    def _prepare_input(self, batch):
        # image = batch["image"]
        # image -= torch.tensor([123.675, 116.28, 103.53], device=self.device).reshape(1, 3, 1, 1)
        # image /= torch.tensor([58.395, 57.12, 57.375], device=self.device).reshape(1, 3, 1, 1)
        # batch["image"] = image
        return super()._prepare_input(batch)
