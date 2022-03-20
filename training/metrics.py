import torch
import torch.nn as nn
from . import base
from . import functional as F
from . import _modules as modules
import torch.distributed as dist

class IoU(base.Metric):
    __name__ = "iou"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, take_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty
        self.take_channels = take_channels

    @torch.no_grad()
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
            take_channels=self.take_channels,
        )


class MicroIoU(base.Metric):
    __name__ = "micro_iou"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-5
        self.intersection = 0.
        self.union = 0.
        self.threshold = threshold

    def reset(self):
        self.intersection = 0.
        self.union = 0.

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction = (prediction > self.threshold).float()

        intersection = (prediction * target).sum()
        union = (prediction + target).sum() - intersection

        self.intersection += intersection.detach()
        self.union += union.detach()

        score = (self.intersection + self.eps) / (self.union + self.eps)
        return score


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt


class MicroF1(base.Metric):
    __name__ = "micro_f1"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-5
        self.threshold = threshold

        self.tp = 0.
        self.gt_count = 0.
        self.pre_count = 0.
        self.score = 0.

    def reset(self, distribe):
        self.tp = 0.
        self.gt_count = 0.
        self.pre_count = 0.
        self.distribe = distribe

    @torch.no_grad()
    def __call__(self, prediction, target):
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[-1]
        prediction = (prediction > self.threshold).float()
        target = nn.functional.interpolate(target,
                                         scale_factor=prediction.size()[2] / target.size()[2],
                                         mode='nearest')

        if self.distribe:
            tp = (prediction * target).sum().detach()
            gt_count = target.sum().detach()
            pre_count = prediction.sum().detach()

            torch.distributed.barrier()
            self.tp += reduce_tensor(tp)
            self.gt_count += reduce_tensor(gt_count)
            self.pre_count += reduce_tensor(pre_count)
        else:
            self.tp += (prediction * target).sum().detach()
            self.gt_count += target.sum().detach()
            self.pre_count += prediction.sum().detach()

        precision = self.tp / (self.pre_count + 1e-8)
        recall = self.tp / (self.gt_count + 1e-8)

        self.score = 2*precision*recall/(precision + recall +  1e-8)
        return self.score

class MicroF1_single(base.Metric):
    __name__ = "micro_f1"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-5
        self.threshold = threshold

        self.tp = 0.
        self.gt_count = 0.
        self.pre_count = 0.

    def reset(self):
        self.tp = 0.
        self.gt_count = 0.
        self.pre_count = 0.

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction = (prediction > self.threshold).float()
        self.tp += (prediction * target).sum().detach()
        self.gt_count += target.sum().detach()
        self.pre_count += prediction.sum().detach()

        precision = self.tp / (self.pre_count + 1e-8)
        recall = self.tp / (self.gt_count + 1e-8)

        score = 2*precision*recall/(precision + recall +  1e-8)
        return score


class MeanIoU(base.Metric):
    __name__ = "mean_iou"

    def __init__(self, ignore_label=0):
        super().__init__()
        self.eps = 1e-5
        self.intersection = {}
        self.union = {}
        self.ignore_label = ignore_label

    def reset(self):
        self.intersection = {}
        self.union = {}

    @torch.no_grad()
    def __call__(self, prediction, target):
        rng = prediction.shape[1]

        prediction = torch.argmax(torch.nn.functional.softmax(prediction, dim=1), dim=1)

        for index in range(rng):
            if index==self.ignore_label:continue
            pre_single = torch.zeros(prediction.shape).float()
            pre_single[prediction == index] = 1.

            gt_single = torch.zeros(target.shape).float()
            gt_single[target == index] = 1.

            intersection = (pre_single * gt_single).sum()

            union = (pre_single + gt_single).sum() - intersection
            if (index in self.intersection) is False:
                self.intersection[index] = 0
            if (index in self.union) is False:
                self.union[index] = 0

            self.intersection[index] += intersection.detach()
            self.union[index] += union.detach()

        score = 0
        for (k, v) in self.intersection.items():
            intersection = self.intersection[k]
            union = self.union[k]
            score += (intersection + self.eps) / (union + self.eps)
        return score/rng




class MIOU(base.Metric):
    __name__ = "miou"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-8
        self.threshold = threshold

        self.tp = 0.
        self.tn = 0.
        self.fn = 0.
        self.fp = 0.

    def reset(self):
        self.tp = 0.
        self.tn = 0.
        self.fn = 0.
        self.fp = 0.

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        return rt

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction = (prediction > self.threshold).float()

        tpt = (prediction * target).sum().detach()
        tnt = ((prediction == 0) * (target == 0)).sum().detach()
        fnt = ((prediction == 0) * (target == 1)).sum().detach()
        fpt = ((prediction == 1) * (target == 0)).sum().detach()

        torch.distributed.barrier()
        self.tp += self.reduce_tensor(tpt)
        self.tn += self.reduce_tensor(tnt)
        self.fn += self.reduce_tensor(fnt)
        self.fp += self.reduce_tensor(fpt)
        iou_0 = self.tp / (self.tp + self.fp + self.fn + self.eps)
        iou_1 = self.tn / (self.tn + self.fp + self.fn + self.eps)
        mIou = 0.5 * iou_0 + 0.5 * iou_1
        return mIou



class MIOU_single(base.Metric):
    __name__ = "micro_f1"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-8
        self.threshold = threshold

        self.tp = 0.
        self.tn = 0.
        self.fn = 0.
        self.fp = 0.

    def reset(self):
        self.tp = 0.
        self.tn = 0.
        self.fn = 0.
        self.fp = 0.

    @torch.no_grad()
    def __call__(self, prediction, target):
        prediction = (prediction > self.threshold).float()
        self.tp += (prediction * target).sum().detach()
        self.tn += ((prediction == 0) * (target == 0)).sum().detach()
        self.fn += ((prediction == 0) * (target == 1)).sum().detach()
        self.fp += ((prediction == 1) * (target == 0)).sum().detach()
        mIou = 0.5*self.tp/(self.tp + self.fp + self.fn + self.eps) + 0.5*self.tn/(self.tn + self.fp + self.fn + self.eps)
        return mIou
