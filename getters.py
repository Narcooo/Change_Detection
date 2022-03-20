import torch.nn as nn
import segmentation_models_pytorch as smp
from training import losses, metrics, optimizers, callbacks
import datasets
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR, ReduceLROnPlateau, _LRScheduler
import torch
# from core.mmseg.mmseg_getter import get_mmseg_model, mmseg_contain
from core.mmodel.mmodel_getter import get_mymodel, mymodel_contain

class ReduceLROnPlateauPatch(ReduceLROnPlateau, _LRScheduler):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]

def get_model(architecture, init_params):
    init_params = init_params or {}
    if mymodel_contain(architecture):
        return get_mymodel(architecture, **init_params)
    else:
        print(architecture)
        model_class = smp.__dict__[architecture]
        return model_class(**init_params)


def get_dataset(name, init_params, **kwargs):
    dataset_class = datasets.__dict__[name]
    dataset = dataset_class(**init_params, **kwargs)
    return dataset


def get_loss(name, init_params):
    init_params = init_params or {}
    loss_class = losses.__dict__[name]
    return loss_class(**init_params)


def get_metric(name, init_params):
    init_params = init_params or {}
    metric_class = metrics.__dict__[name]
    return metric_class(**init_params)


def get_optimizer(name, model_params, init_params):
    assert init_params is not None
    if name=='SGD':
        return torch.optim.SGD(model_params, **init_params)

    optim_class = optimizers.__dict__[name]
    # TODO: make parsing of model parameters for different LR
    return optim_class(model_params, **init_params)


def get_scheduler(name, optimizer, init_params):
    init_params = init_params or {}
    if name=='cosineAnnWarm':
        # T_0=5,T_mult=2
        scheduler = CosineAnnealingWarmRestarts(optimizer, **init_params)
    elif name == 'ReduceLROnPlateau':
        # T_0=5,T_mult=2
        print("=================================")
        print(init_params)
        print("=================================")
        scheduler = ReduceLROnPlateauPatch(optimizer, **init_params)
    else:
        scheduler_class = optimizers.__dict__[name]
        scheduler = scheduler_class(optimizer, **init_params)
    return scheduler


def get_callback(name, init_pararams):
    init_pararams = init_pararams or {}
    callback_class = callbacks.__dict__[name]
    callback = callback_class(**init_pararams)
    return callback
