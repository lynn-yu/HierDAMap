import os.path

from einops import rearrange
from torch.nn import functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import torchvision
from Our.loss_edl import *
def vis_mapping(semantic,):
    plt.figure()
    #plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.imshow(semantic[1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
    plt.imshow(semantic[2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
    plt.imshow(semantic[3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
    plt.show()
def show(img,name='fig',save=False,path=None,number=0):
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    if not save:
        plt.title(name)
        plt.show()
    else:
        path_name = os.path.join(path,str(number)+'_'+name)
        plt.savefig(path_name)

def mask_mix(mb_x,x_bev_init,cross_bev,i):
    if mb_x == 0:
        x_t = x_bev_init[i, :, :, 0:100].clone()
        x_s = cross_bev[i, :, :, 100:200].clone()
        x_mix = torch.cat((x_t, x_s), dim=2)
    if mb_x == 1:
        x_t = x_bev_init[i, :, :, 100:200].clone()
        x_s = cross_bev[i, :, :, 0:100].clone()
        x_mix = torch.cat((x_s, x_t), dim=2)
    elif mb_x == 2:
        x_t = x_bev_init[i, :, 0:100, :].clone()
        x_s = cross_bev[i, :, 100:200, :].clone()
        x_mix = torch.cat((x_t, x_s), dim=1)
    elif mb_x == 3:
        x_t = x_bev_init[i, :, 100:200, :].clone()
        x_s = cross_bev[i, :, 0:100, :].clone()
        x_mix = torch.cat((x_s, x_t), dim=1)
    return x_mix




def change_view(x,y,mb):
    b = mb.shape[0]
    d = torch.linspace(0, b - 1, b)
    x[d.long(), mb.long()] = y[d.long(), mb.long()]
    return x

def sigmoid_rampup(current, rampup_length):
                """Exponential rampup from https://arxiv.org/abs/1610.02242"""
                if rampup_length == 0:
                    return 1.0
                else:
                    current = np.clip(current, 0.0, rampup_length)
                    phase = 1.0 - current / rampup_length
                    return float(np.exp(-5.0 * phase * phase))
def get_single_mask_map(vertices):
    tri_mask1 = np.zeros((400, 200))
    #vertices = np.array([[30, 0], [100, 100], [170, 0]], np.int32)
    pts = vertices.reshape((-1, 1, 2))
    cv2.fillPoly(tri_mask1, [pts], color=1)
    #cv2.circle(tri_mask1, (100, 100), 10, color=0, thickness=-1)
    return tri_mask1

def get_view_mask_short_map(cam=6):

    none_mask = np.zeros((400, 200))
    short_mask = get_single_mask_map(vertices=np.array([[50,100], [150, 100], [150, 300], [50, 300]], np.int32))#get_short_mask()
    front_mask = short_mask * get_single_mask_map(vertices=np.array([[0,342], [0,400],[200,400],[200,342], [100,200]], np.int32))
    front_l_mask = short_mask * get_single_mask_map(vertices = np.array([[172,400],[200,400], [200, 200], [100,200]], np.int32))
    front_r_mask = short_mask * get_single_mask_map(vertices = np.array([[28,400],[0,400], [0,200], [100,200]], np.int32))
    back_mask = short_mask * get_single_mask_map(vertices = np.array([[0,130],[0,0], [200, 0], [200,130],[100,200]], np.int32))
    back_l_mask = short_mask * get_single_mask_map(vertices = np.array([[200,58],[100,200],[200,216]], np.int32))
    back_r_mask = short_mask * get_single_mask_map(vertices = np.array([[0,58],[100,200],[0,216]], np.int32))
    if cam==6:
        a = [front_l_mask,front_mask,front_r_mask,back_l_mask,back_mask,back_r_mask]
    else:
        a = [front_mask,front_l_mask, front_mask, front_r_mask, back_l_mask, back_mask, back_r_mask]

    return torch.tensor(a)

def get_mask(mask, mb):
    b = mb.shape[0]
    mask = mask.unsqueeze(0).repeat(b, 1, 1, 1)
    d = torch.linspace(0, b - 1, b)
    mask_b = mask[d.long(), mb.long()]  # b h w
    mask_b = mask_b.unsqueeze(1).float().cuda()
    return mask_b
def get_single_mask(vertices):
    tri_mask1 = np.zeros((200, 200))
    #vertices = np.array([[30, 0], [100, 100], [170, 0]], np.int32)
    pts = vertices.reshape((-1, 1, 2))
    cv2.fillPoly(tri_mask1, [pts], color=1)
    #cv2.circle(tri_mask1, (100, 100), 10, color=0, thickness=-1)
    return tri_mask1
def get_short_mask():
    tri_mask1 = np.zeros((200, 200))
    cv2.circle(tri_mask1, (100, 100), 50, color=1, thickness=-1)
    return tri_mask1
def get_view_mask_short():
    none_mask = np.zeros((200, 200))
    short_mask = get_single_mask(vertices=np.array([[50,50], [150, 50], [150, 150], [50, 150]], np.int32))#get_short_mask()
    front_mask = short_mask * get_single_mask(vertices=np.array([[30, 200], [100, 100], [170, 200]], np.int32))
    front_l_mask = short_mask * get_single_mask(vertices = np.array([[200,200],[136,200], [100, 100], [200,100]], np.int32))
    front_r_mask = short_mask * get_single_mask(vertices = np.array([[0,200],[64,200], [100, 100], [0,100]], np.int32))
    back_mask = short_mask * get_single_mask(vertices = np.array([[0,30],[0,0], [200, 0], [200,30],[100,100]], np.int32))
    back_l_mask = short_mask * get_single_mask(vertices = np.array([[170,0],[100,100],[200,126], [200,0]], np.int32))
    back_r_mask = short_mask * get_single_mask(vertices = np.array([[30,0],[100,100],[0,126], [0,0]], np.int32))
    a = [front_l_mask,front_mask,front_r_mask,back_l_mask,back_mask,back_r_mask]

    return torch.tensor(a)
def get_view_mask():
    none_mask = np.zeros((200, 200))
    front_mask = get_single_mask(vertices=np.array([[30, 200], [100, 100], [170, 200]], np.int32))
    front_l_mask = get_single_mask(vertices = np.array([[200,200],[136,200], [100, 100], [200,100]], np.int32))
    front_r_mask = get_single_mask(vertices = np.array([[0,200],[64,200], [100, 100], [0,100]], np.int32))
    back_mask = get_single_mask(vertices = np.array([[0,30],[0,0], [200, 0], [200,30],[100,100]], np.int32))
    back_l_mask = get_single_mask(vertices = np.array([[170,0],[100,100],[200,126], [200,0]], np.int32))
    back_r_mask = get_single_mask(vertices = np.array([[30,0],[100,100],[0,126], [0,0]], np.int32))
    a = [front_l_mask,front_mask,front_r_mask,back_l_mask,back_mask,back_r_mask]
    return torch.tensor(a)



def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def sigmoid_rampdown(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6,wramup=0):
        # super(PolyLR,self).__init__(optimizer,last_epoch)
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.wramup = wramup
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - (self.last_epoch - self.wramup) / (self.max_iters - self.wramup)) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
    "car":(202, 178, 214),
}

def visual(masks,classes=['road_segment','ped_crossing', 'walkway', 'stop_line','carpark_area','divider','car',],background=(255,255,255),save=False,names=None,path=None):

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background
    n = masks.shape[0]
    for k, name in enumerate(classes):
        if k<n:
            if name in MAP_PALETTE:
                canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    plt.figure()
    plt.axis('off')
    plt.imshow(canvas)
    if save:
        pathname = os.path.join(path, names)
        plt.savefig(pathname)
    else:
        plt.show()
    plt.close()

class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
def vis_hdmap(masks,save=False,names=None,path=None,classes=['road_segment','ped_crossing','divider'],background=(255,255,255)):
    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background
    n = masks.shape[0]
    for k, name in enumerate(classes):
        if k < n:
            if name in MAP_PALETTE:
                canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    plt.figure()
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.imshow(canvas)
    if save:
        pathname = os.path.join(path, names)
        plt.savefig(pathname)
    else:
        plt.show()

def vis_img(image,save=False,name=None,path=None):
    denormalize_img = torchvision.transforms.Compose((
        NormalizeInverse(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ToPILImage(),
    ))
    imgs = [denormalize_img(image[i]) for i in range(image.shape[0])]
    a = np.hstack(imgs[:3])
    b = np.hstack(imgs[3:])
    vi = np.vstack((a, b))
    plt.figure()
    plt.axis('off')
    plt.imshow(vi)
    if save:
        pathname = os.path.join(path,name)
        plt.savefig(pathname)
    else:
        plt.show()

def vis_pv_mask(image):

    imgs = [image[i][1] for i in range(image.shape[0])]
    a = np.hstack(imgs[:3])
    b = np.hstack(imgs[3:])
    vi = np.vstack((a, b))
    plt.figure()
    plt.axis('off')
    plt.imshow(vi)
    plt.show()

def vis_fea(f):
    f = torch.sum(f,dim=0)
    plt.figure()
    plt.imshow(f)
    plt.show()

class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False,
                 top_k_ratio=1.0, future_discount=1.0,weight=1.0):

        super().__init__()

        self.class_weights = torch.tensor(class_weights).float()
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount
        self.weight = weight

    def forward(self, prediction, target):
        b,c, h, w = prediction.shape
        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=self.class_weights.to(target.device).float(),
        )

        loss = loss.view(b,  -1)
        m = loss.mean()
        return self.weight*m

def visual_fea(fea):
    plt.figure(figsize=(30, 30))
    c,h,w = fea.shape
    for i in range(c):
        if i == 4:  # 只生成64张特征图
            break
        plt.subplot(2, 2, i+1)
        plt.imshow(fea[i], cmap='gray')
        plt.axis('off')
    plt.show()

def onehot_encoding_ori(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot
def onehot_encoding(logits, dim=1):
    logits = logits.sigmoid()
    logits = logits>0.5
    return logits

def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = preds.bool()
        gt_map = binimgs.bool()
        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)
    intersects, unions =  torch.tensor(intersects), torch.tensor(unions)
    return intersects, unions, intersects / (unions  + 1e-7)

def get_batch_iou_bev(pred,label):
    thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]).cuda()

    num_classes = pred.shape[1]
    num_thresholds = len(thresholds)

    tp = torch.zeros(num_classes, num_thresholds).cuda()
    fp = torch.zeros(num_classes, num_thresholds).cuda()
    fn = torch.zeros(num_classes, num_thresholds).cuda()


    pred = pred.detach().reshape(num_classes, -1)
    label = label.detach().bool().reshape(num_classes, -1)

    pred = pred[:, :, None] >= thresholds
    label = label[:, :, None]

    tp += (pred & label).sum(dim=1)
    fp += (pred & ~label).sum(dim=1)
    fn += (~pred & label).sum(dim=1)

    ious = tp / (tp + fp + fn + 1e-7)
    ious = ious.max(dim=1).values.mean()
    return ious

class Dice_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid_func = nn.Sigmoid()

    def forward(self, logits, labels, is_sigmoid=False,*args):  # B x 14 x H x W, B x 14 x H x W
        if not  is_sigmoid:
            logits = self.sigmoid_func(logits)  # B x 14 x H x W, 0~1
        # labels = labels.float()  # B x 14 x H x W, 0/1

        intersection = 2 * logits * labels  # B x 14 x H x W
        union = (logits + labels)  # B x 14 x H x W

        intersection = intersection.sum(dim=0).sum(dim=-1).sum(dim=-1)  # 14
        union = union.sum(dim=0).sum(dim=-1).sum(dim=-1)  # 14

        iou = intersection / (union + 1e-5)  # 14

        dice_loss = 1 - iou.mean()

        return dice_loss


class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def forward(self, ystudent, yteacher):
        loss = self.loss_fn(ystudent, yteacher)
        return loss
class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight=2.13):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss

class DomainLoss(torch.nn.Module):
    def __init__(self):
        super(DomainLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_label):
        y_truth_tensor = torch.FloatTensor(y_pred.size())
        y_truth_tensor.fill_(y_label)
        y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
        return self.loss_fn(y_pred, y_truth_tensor)


class DepthLoss(torch.nn.Module):
    def __init__(self, depth_weight=None, gt_type='probability'):
        super(DepthLoss, self).__init__()
        self.gt_type = gt_type
        self.depth_weight = depth_weight

    def cross_entropy(self, pred, soft_targets):
        if self.depth_weight is None:
            return torch.mean(torch.sum(-soft_targets * torch.log(pred), 1))
        else:
            return torch.mean(torch.matmul(-soft_targets * torch.log(pred), self.depth_weight))

    def filter_empty(self, depth, lidars):
        # lidars, depth: [B, N, final_dim[0]//downsample, final_dim[1]//downsample, D]
        _, _, _, _, D = lidars.shape
        lidars = lidars.reshape(-1, D)
        depth = depth.reshape(-1, D)

        # filter pixels without points
        mask = torch.logical_and(torch.sum(lidars, 1) > 0, torch.min(depth, 1)[0] > 0)
        lidars = lidars[mask, :]
        depth = depth[mask, :]

        return depth, lidars

    def forward(self, depth, lidars):
        # lidars, depth: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        depth = depth.permute(0, 1, 3, 4, 2)
        lidars = lidars.permute(0, 1, 3, 4, 2)

        depth, lidars = self.filter_empty(depth, lidars)
        if self.gt_type == 'probability':
            lidars = lidars / lidars.sum(dim=1).view(-1, 1)
        else:  # Change to other ways here
            lidars = lidars / lidars.sum(dim=1).view(-1, 1)

        loss = self.cross_entropy(depth, lidars)

        return loss, torch.isnan(loss).item()

def py_sigmoid_focal_loss(pred,
                          target,
                          one_hot_target=None,
                          weight=None,
                          gamma=2.0,
                          alpha=0.5,
                          class_weight=None,
                          valid_mask=None,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction with
            shape (N, C)
        one_hot_target (None): Placeholder. It should be None.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float | list[float], optional): A balanced form for Focal Loss.
            Defaults to 0.5.
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        valid_mask (torch.Tensor, optional): A mask uses 1 to mark the valid
            samples and uses 0 to mark the ignored samples. Default: None.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    if isinstance(alpha, list):
        alpha = pred.new_tensor(alpha)
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * one_minus_pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    final_weight = torch.ones(1, pred.size(1)).type_as(loss)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       one_hot_target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.5,
                       class_weight=None,
                       valid_mask=None,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction. It's shape
            should be (N, )
        one_hot_target (torch.Tensor): The learning label with shape (N, C)
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float | list[float], optional): A balanced form for Focal Loss.
            Defaults to 0.5.
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        valid_mask (torch.Tensor, optional): A mask uses 1 to mark the valid
            samples and uses 0 to mark the ignored samples. Default: None.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    final_weight = torch.ones(1, pred.size(1)).type_as(pred)
    if isinstance(alpha, list):
        # _sigmoid_focal_loss doesn't accept alpha of list type. Therefore, if
        # a list is given, we set the input alpha as 0.5. This means setting
        # equal weight for foreground class and background class. By
        # multiplying the loss by 2, the effect of setting alpha as 0.5 is
        # undone. The alpha of type list is used to regulate the loss in the
        # post-processing process.
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                   gamma, 0.5, None, 'none') * 2
        alpha = pred.new_tensor(alpha)
        final_weight = final_weight * (
            alpha * one_hot_target + (1 - alpha) * (1 - one_hot_target))
    else:
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                   gamma, alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.5,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_focal'):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal
                Loss. Defaults to 0.5. When a list is provided, the length
                of the list should be equal to the number of classes.
                Please be careful that this parameter is not the
                class-wise weight but the weight of a binary classification
                problem. This binary classification problem regards the
                pixels which belong to one class as the foreground
                and the other pixels as the background, each element in
                the list is the weight of the corresponding foreground class.
                The value of alpha or each element of alpha should be a float
                in the interval [0, 1]. If you want to specify the class-wise
                weight, please use `class_weight` parameter.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_focal'.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, \
            'AssertionError: Only sigmoid focal loss supported now.'
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        assert isinstance(class_weight, list) or class_weight is None, \
            'AssertionError: class_weight must be None or of type list'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape
                (N, C) where C = number of classes, or
                (N, C, d_1, d_2, ..., d_K) with K≥1 in the
                case of K-dimensional loss.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1,
                or (N, d_1, d_2, ..., d_K) with K≥1 in the case of
                K-dimensional loss. If containing class probabilities,
                same shape as the input.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used
                to override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            ignore_index (int, optional): The label index to be ignored.
                Default: 255
        Returns:
            torch.Tensor: The calculated loss
        """
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
               "The shape of pred doesn't match the shape of target"

        original_shape = pred.shape

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            # target with shape [B, C, d_1, d_2, ...]
            # transform it's shape into [N, C]
            # [B, C, d_1, d_2, ...] -> [C, B, d_1, d_2, ..., d_k]
            target = target.transpose(0, 1)
            # [C, B, d_1, d_2, ..., d_k] -> [C, N]
            target = target.reshape(target.size(0), -1)
            # [C, N] -> [N, C]
            target = target.transpose(0, 1).contiguous()
        else:
            # target with shape [B, d_1, d_2, ...]
            # transform it's shape into [N, ]
            target = target.view(-1).contiguous()
            valid_mask = (target != ignore_index).view(-1, 1)
            # avoid raising error when using F.one_hot()
            target = torch.where(target == ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            if torch.cuda.is_available() and pred.is_cuda:
                if target.dim() == 1:
                    one_hot_target = F.one_hot(target, num_classes=num_classes)
                else:
                    one_hot_target = target
                    target = target.argmax(dim=1)
                    valid_mask = (target != ignore_index).view(-1, 1)
                calculate_loss_func = sigmoid_focal_loss
            else:
                one_hot_target = None
                if target.dim() == 1:
                    target = F.one_hot(target, num_classes=num_classes)
                else:
                    valid_mask = (target.argmax(dim=1) != ignore_index).view(
                        -1, 1)
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                one_hot_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                class_weight=self.class_weight,
                valid_mask=valid_mask,
                reduction=reduction,
                avg_factor=avg_factor)

            if reduction == 'none':
                # [N, C] -> [C, N]
                loss_cls = loss_cls.transpose(0, 1)
                # [C, N] -> [C, B, d1, d2, ...]
                # original_shape: [B, C, d1, d2, ...]
                loss_cls = loss_cls.reshape(original_shape[1],
                                            original_shape[0],
                                            *original_shape[2:])
                # [C, B, d1, d2, ...] -> [B, C, d1, d2, ...]
                loss_cls = loss_cls.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
