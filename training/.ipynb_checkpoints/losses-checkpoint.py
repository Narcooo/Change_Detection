import torch
import torch.nn as nn

from . import base
from . import functional as F
from . import _modules as modules
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from . import lovasz_losses as L
from . import flip_loss


class ftnmt_loss(base.Loss):

    def __init__(self, depth=5, axis=[1, 2, 3], smooth=1.0e-5, **kwargs):
        super().__init__(**kwargs)

        assert depth >= 0, ValueError("depth must be >= 0, aborting...")

        self.smooth = smooth
        self.axis = axis
        self.depth = depth

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1. / depth

    def inner_prod(self, prob, label):
        prod = torch.mul(prob, label)
        prod = torch.sum(prod, axis=self.axis)
        return prod

    def tnmt_base(self, preds, labels):
        tpl = self.inner_prod(preds, labels)
        tpp = self.inner_prod(preds, preds)
        tll = self.inner_prod(labels, labels)

        num = tpl + self.smooth
        scale = 1. / self.depth
        denum = 0.0
        for d in range(self.depth):
            a = 2. ** d
            b = -(2. * a - 1.)
            denum = denum + torch.reciprocal(torch.add(a * (tpp + tll), b * tpl) + self.smooth)

        result = torch.mul(num, denum) * scale
        return torch.mean(result, dim=0, keepdim=True)

    def forward(self, preds, labels):
        l1 = self.tnmt_base(preds, labels)
        l2 = self.tnmt_base(1. - preds, 1. - labels)
        result = 0.5 * (l1 + l2)
        return 1. - result


class JaccardLoss(base.Loss):

    def __init__(self, eps=1e-7, activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = modules.Activation(activation, dim=1)
        self.per_image = per_image
        self.ignore_channels = ignore_channels
        self.class_weights = class_weights

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1e-7, beta=1., activation=None, ignore_channels=None,
                 per_image=False, class_weights=None, drop_empty=False, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = modules.Activation(activation, dim=1)
        self.ignore_channels = ignore_channels
        self.per_image = per_image
        self.class_weights = class_weights
        self.drop_empty = drop_empty

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
            per_image=self.per_image,
            class_weights=self.class_weights,
            drop_empty=self.drop_empty,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    def __init__(self, reduce=True, size_average=True):
        super().__init__()


class MSEloss(base.Loss):

    def __init__(self):
        super(MSEloss, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        return self.loss_fn(pre, gt)


class BCELoss(base.Loss):

    def __init__(self, pos_weight=1., neg_weight=1., reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        loss = F.binary_crossentropy(
            pr, gt,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            label_smoothing=self.label_smoothing,
        )

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class BinaryFocalLoss(base.Loss):
    def __init__(self, alpha=1, gamma=2, class_weights=None, logits=False, reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None]
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.class_weights = class_weights if class_weights is not None else 1.
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        if self.logits:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pr, gt, reduction='none')
        else:
            bce_loss = F.binary_crossentropy(pr, gt, label_smoothing=self.label_smoothing)

        pt = torch.exp(- bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss * torch.tensor(self.class_weights).to(focal_loss.device)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()

        return focal_loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class FocalDiceLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.focal = BinaryFocalLoss()
        self.dice = DiceLoss(eps=10.)

    def __call__(self, y_pred, y_true):
        return 2 * self.focal(y_pred, y_true) + self.dice(y_pred, y_true)


class FocalFtLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.focal = BinaryFocalLoss()
        self.ftloss = ftnmt_loss()

    def __call__(self, y_pred, y_true):
        return self.focal(y_pred, y_true) + self.ftloss(y_pred, y_true)


class BCEDiceLoss(base.Loss):

    def __init__(self):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss(eps=10.)

    def __call__(self, y_pred, y_true):
        return 2 * self.bce(y_pred, y_true) + self.dice(y_pred, y_true)


class BBCELoss(base.Loss):

    def __init__(self, reduction='mean', label_smoothing=None):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pr, gt):
        pos = torch.eq(gt, 1).float()
        neg = torch.eq(gt, 0).float()
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg
        alpha_pos = num_neg / num_total
        alpha_neg = num_pos / num_total

        loss = F.binary_crossentropy(
            pr, gt,
            pos_weight=alpha_pos,
            neg_weight=alpha_neg,
            label_smoothing=self.label_smoothing,
        )

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

class Lovaszsigmoid(base.Loss):
    def __init__(self, reduction='mean', ignore_label=255):
        super().__init__()
        assert reduction in ['mean', None, False]
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, pr, gt):
        # Lovasz need sigmoid input
        # out = torch.sigmoid(pr)
        loss = L.lovasz_softmax(pr, gt, classes=[1], ignore=self.ignore_label)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss





#-----------------------------------------------------------------------------------------------------------------------
#
#
class Lovaszsoftmax(nn.Module):
    def __init__(self, ignore_label=0):
        super(Lovaszsoftmax, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, pre, gt):
        pre = torch.nn.functional.softmax(pre, dim=1)
        return L.lovasz_softmax(pre, gt, ignore=self.ignore_label)


class CrossEntropy(base.Loss):
    def __init__(self, ignore_label=0, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, pre, gt):
        ph, pw = pre.size(2), pre.size(3)
        h, w = gt.size(1), gt.size(2)
        if ph != h or pw != w:
            pre = torch.nn.functional.upsample(
                    input=pre, size=(h, w), mode='bilinear')

        loss = self.criterion(pre, gt)

        return loss


#-----------------------------------------------------------------------------------------------------------------------
#
#
class Changeloss(nn.Module):
    def __init__(self):
        super(Changeloss, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt

        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small
        return self.loss_fn(pre1, pre2)*2


class Changeloss_LEVIR(nn.Module):
    def __init__(self):
        super(Changeloss_LEVIR, self).__init__()
        self.loss_fn = FocalDiceLoss()
        self.activation = nn.Sigmoid()
        self.loss_smm = Changeloss()

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)
        pre_union = self.activation(pre[:, :channel] + pre[:, channel:])*gt
        return self.loss_fn(pre_union, gt) + self.loss_smm(pre, gt)


class Changeloss_new(nn.Module):
    def __init__(self):
        super(Changeloss_new, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        self.loss_dff = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt

        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small

        pre1_diff = pre[:, :channel] * gt
        pre2_diff = pre[:, channel:] * gt
        loss = self.loss_fn(pre1, pre2) + (1 - self.loss_dff(pre1_diff, pre2_diff))
        # for pre in pres[1:]:
        #     scale = pre.shape[-1]/gt.shape[-1]
        #     gt_resize = nn.functional.interpolate(gt, scale_factor=scale, mode='nearest', align_corners=None)
        #     channel = int(pre.shape[1] / 2)
        #     gt_small = 1 - gt_resize
        #     pre1 = pre[:, :channel] * gt_small
        #     pre2 = pre[:, channel:] * gt_small
        #     pre1_diff = pre[:, :channel] * gt_resize
        #     pre2_diff = pre[:, channel:] * gt_resize
        #     loss +=self.loss_fn(pre1, pre2) + (1 - self.loss_dff(pre1_diff, pre2_diff))

        return loss

class Changeloss_self(nn.Module):
    def __init__(self):
        super(Changeloss_self, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        self.loss_dff = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt
        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small

        pre1_diff = pre[:, :channel] * gt
        pre2_diff = pre[:, channel:] * gt

        return self.loss_fn(pre1, pre2) + (1 - self.loss_dff(pre1_diff, pre2_diff))


class Changelosswd(nn.Module):
    def __init__(self):
        super(Changelosswd, self).__init__()
        self.ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=7)
        # self.WDLoss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        # self.loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt

        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small
        p_output = torch.nn.functional.softmax(pre1, dim=1)
        q_output = torch.nn.functional.softmax(pre2, dim=1)
        return 1- self.ssim_loss(p_output, q_output)



class Changelossflip(nn.Module):
    def __init__(self, useactive = False):
        super(Changelossflip, self).__init__()
        self.flip_loss = flip_loss.FLIPLoss()
        self.useactive = useactive

    def forward(self, pre, gt):
        channel = int(pre.shape[1] / 2)

        gt_small = 1 - gt

        pre1 = pre[:, :channel] * gt_small
        pre2 = pre[:, channel:] * gt_small
        if self.useactive:
            pre1 = torch.nn.functional.softmax(pre1, dim=1)
            pre2 = torch.nn.functional.softmax(pre2, dim=1)
        return self.flip_loss(pre1, pre2)


class contrastive_loss(nn.Module):
    def __init__(self, tau=1, normalize=False):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss