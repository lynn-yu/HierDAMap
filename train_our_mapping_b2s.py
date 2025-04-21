import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import time
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from data.nus_mapping import compile_data_mapping
from Our.LSS_model_da import LiftSplatShoot_mask_mix
from tools import *
from torch.optim import SGD, AdamW, Adam
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from Our.confusion import BinaryConfusionMatrix, singleBinary

score_th = 0.5

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

        # ce_loss = self.ce_criterion(prediction, target)
        # pred_logsoftmax = F.log_softmax(prediction)
        # loss = self.nll_criterion(pred_logsoftmax, target)

        loss = loss.view(b,  -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]
        m = loss.mean()
        return self.weight*m

def onehot_encoding_mapping(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def get_batch_iou_mapping(preds, binimgs):
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

def get_val_info(model, valloader, device, use_tqdm=True):

    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    confusion_c = BinaryConfusionMatrix(1)
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs,_ = batch

            preds, _, _, _, _, _, pred_car = model(allimgs.to(device), rots.to(device),trans.to(device), intrins.to(device),
                         post_rots.to(device),post_trans.to(device), lidars.to(device), )

            # iou
            intersect, union, _ = get_batch_iou_mapping(onehot_encoding_mapping(preds),  binimgs['iou'].cuda().long())
            total_intersect += intersect
            total_union += union
            scores_c = pred_car.cpu().sigmoid()
            confusion_c.update(scores_c > score_th, (binimgs['car']) > 0)


    iou = total_intersect / (total_union+1e-7)
    miou = torch.mean(iou[1:])
    return {'iou': iou,'miou': miou,'miou_car': confusion_c.mean_iou}


def train(logdir, grid_conf, data_aug_conf, version, dataroot, nsweeps, domain_gap, source, target, bsz, nworkers, lr, weight_decay, nepochs,
          max_grad_norm=5.0, gpuid=0, ):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logging.basicConfig(filename=os.path.join(logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Lidar Sup + Mini BEV Sup")
    strainloader, ttrainloader, tvalloader = compile_data_mapping(version, dataroot, data_aug_conf, grid_conf, nsweeps, domain_gap, source, target, bsz,
                                                                 nworkers, flip=True)

    straingenerator = iter(strainloader)
    ttraingenerator = iter(ttrainloader)
    if len(strainloader) < len(ttrainloader):
        datalen = len(ttrainloader)
    else:
        datalen = len(strainloader)

    device = torch.device(f'cuda:{gpuid}')

    student_model = LiftSplatShoot_mask_mix(grid_conf, data_aug_conf, outC=4)
    teacher_model = LiftSplatShoot_mask_mix(grid_conf, data_aug_conf, outC=4)
    student_model.to(device)
    teacher_model.to(device)
    epoch = int(nepochs)


    depth_weights = torch.Tensor([2.4, 1.2, 1.0, 1.1, 1.2, 1.4, 1.8, 2.3, 2.7, 3.5,
                                  3.6, 3.9, 4.8, 5.8, 5.4, 5.3, 5.0, 5.4, 5.3, 5.9,
                                  6.5, 7.0, 7.5, 7.5, 8.5, 10.3, 10.9, 9.8, 11.5, 13.1,
                                  15.1, 15.1, 16.3, 16.3, 17.8, 19.6, 21.8, 24.5, 24.5, 28.0,
                                  28.0]).to(device)
    loss_depth = DepthLoss(depth_weights).cuda(gpuid)

    loss_pv = Dice_Loss().cuda()
    loss_final = SegmentationLoss(class_weights=[1.0, 2.0, 2.0, 2.0], weight=1.0).cuda()
    loss_mini = Dice_Loss().cuda()
    loss_car = Dice_Loss().cuda()

    opt = AdamW(student_model.parameters(), lr=lr)
    start = 0.7
    # sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=datalen, epochs=epoch,anneal_strategy='cos',
    #                                             pct_start=start, div_factor=2,final_div_factor=3)
    sched = StepLR(opt, 10, 0.1)
    uplr = int(datalen * epoch * 0.6)
    logger.info(f"sigmoid step: {uplr}  ")
    wp = 0.5
    logger.info(f'weight pv: {wp} ')

    student_model.train()
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.detach_()

    loss_nan_counter = 0
    np.random.seed()

    for epo in range(1, epoch):
        iteration = (epo - 1) * datalen
        for i in range(datalen):
            t0 = time.time()
            ###source student
            try:
                imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s = next(straingenerator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                straingenerator = iter(strainloader)
                imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s = next(straingenerator)

            seg_mask = rearrange(seg_mask_s, 'b n c h w -> (b n) c h w')
            lidars_i = rearrange(lidars, 'b n c h w -> (b n) c h w')
            preds, depth, pv_out, x_bev, x_mini, mask_mini, pred_car = student_model(imgs.to(device), rots.to(device),
                                                                                     trans.to(device), intrins.to(device),
                                                                                     post_rots.to(device),
                                                                                     post_trans.to(device),
                                                                                     lidars_i.to(device), camdroup=False,
                                                                                     seg_mask=seg_mask.to(device), )

            binimgs = binimgs_s['index'].to(device)
            binimgs_car = binimgs_s['car'].to(device).unsqueeze(1)
            loss_f = loss_final(preds, binimgs)
            loss_f_car = loss_car(pred_car, binimgs_car)
            loss_p = wp * loss_pv(pv_out, seg_mask.cuda())
            loss_d, d_isnan = loss_depth(depth, lidars.to(device))
            loss_d = 0.1 * loss_d
            # visual((binimgs[0] > 0.2).detach().cpu())
            # visual((preds.sigmoid()[0] > 0.2).detach().cpu())
            ###target
            try:
                un_image, un_rots, un_trans, un_intrins, un_post_rots, un_post_trans, un_lidars, un_binimgs_t, \
                    un_img_ori, un_post_rots_ori, un_post_trans_ori, seg_mask_t, seg_mask_o = next(ttraingenerator)

            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                ttraingenerator = iter(ttrainloader)
                un_image, un_rots, un_trans, un_intrins, un_post_rots, un_post_trans, un_lidars, un_binimgs_t, \
                    un_img_ori, un_post_rots_ori, un_post_trans_ori, seg_mask_t, seg_mask_o = next(ttraingenerator)
            #visual(binimgs_s[0]>0.5)
            un_binimgs = un_binimgs_t['index'].to(device)
            un_binimgs_car = un_binimgs_t['car'].to(device).unsqueeze(1)
            with torch.no_grad():
                preds_un_ori, _, _, x_bev_un_ori, _, mask_mini_un_ori, pred_car_ori = teacher_model(un_img_ori.to(device),
                                                                                                un_rots.to(device),
                                                                                                un_trans.to(device),
                                                                                                un_intrins.to(device),
                                                                                                un_post_rots_ori.to(device),
                                                                                                un_post_trans_ori.to(device),
                                                                                                None, )
            seg_mask_un = rearrange(seg_mask_t, 'b n c h w -> (b n) c h w')
            preds_un, _, pv_out_un, x_bev_un, x_mini_un, mask_mini_un, pred_car_un = student_model(un_image.to(device), un_rots.to(device),
                                                                                                   un_trans.to(device), un_intrins.to(device),
                                                                                                   un_post_rots.to(device), un_post_trans.to(device), None,
                                                                                                   camdroup=False, seg_mask=seg_mask_un.to(device),
                                                                                                   )  #



            loss_p_un = wp * loss_pv(pv_out_un, seg_mask_un.cuda())

            it = i + (epo - 1) * datalen
            weightofcon = sigmoid_rampup(it, uplr)
            w1 = min(0.002 + 0.1 * weightofcon, 0.1)
            loss_bev = 0.1 * w1 * torch.square((x_bev_un - x_bev_un_ori)).mean() \
                       + w1 * torch.square((preds_un.sigmoid() - preds_un_ori.sigmoid())).mean() + w1 * torch.square(pred_car_un.sigmoid() - pred_car_ori.sigmoid()).mean()



            ###mini bev
            x_mini = torch.cat((x_mini, x_mini_un, x_mini_droup), dim=1)  #
            mask_mini = torch.cat((mask_mini, mask_mini_un, mask_mini_droup), dim=1)  # ,mask_mini_un_ori
            loss_mbs = 0.1 * w1 * loss_mini(x_mini, mask_mini)


            opt.zero_grad()

            loss = loss_p + loss_p_un + loss_bev + loss_f + loss_f_car + 0.1*loss_mbs
            if d_isnan:
                loss_nan_counter += 1
            else:
                loss = loss + loss_d

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            opt.step()

            alpha = min(1 - 1 / (iteration + 1), 0.999)  # when iteration>=999, alpha==0.999
            with torch.no_grad():
                model_state_dict = student_model.state_dict()
                ema_model_state_dict = teacher_model.state_dict()
                for entry in ema_model_state_dict.keys():
                    ema_param = ema_model_state_dict[entry].clone().detach()
                    param = model_state_dict[entry].clone().detach()
                    new_param = (ema_param * alpha) + (param * (1. - alpha))
                    ema_model_state_dict[entry] = new_param
                teacher_model.load_state_dict(ema_model_state_dict)


            if i > 0 and i % 10 == 0:
                _, _, iou = get_batch_iou_mapping(onehot_encoding_mapping(preds), binimgs_s['iou'].cuda())
                _,_,iou_un = get_batch_iou_mapping(onehot_encoding_mapping(preds_un_ori), un_binimgs_t['iou'].cuda(), )
                iou_car, miou_car = singleBinary(pred_car_ori.sigmoid() > score_th, un_binimgs_car > 0, dim=1)

                logger.info(f"EVAL[{int(epo):>3d}]: [{i:>4d}/{datalen}]:    "
                            f"lr {opt.param_groups[0]['lr']:>.2e}   "
                            f"w1 {w1:>7.4f}   "
                            f"loss: {loss.item():>7.4f}   "
                            #f"loss_f_car: {loss_f_car.item():>7.4f}   "
                            f"loss_bev: {loss_bev.item():>7.4f}   "
                            # f"loss_bev_d: {loss_bev_d.item():>7.4f}   "
                            # f"loss_mbs: {loss_mbs:>7.4f}   "
                            #f"loss_bev_mix: {loss_bev_mix.item():>7.4f}   "
                            #f"mIoU: {torch.mean(iou[1:]):>7.4f}  "
                            f"mIOU_car: {miou_car:>7.4f}  "
                            f"tIOU: {np.array2string(iou[1:].cpu().numpy(), precision=3, floatmode='fixed')}  "
                            f"IOU_un: {np.array2string(iou_un[1:].cpu().numpy(), precision=3, floatmode='fixed')}  "

                            )

            iteration += 1
        sched.step()
        if True:
            val_info = get_val_info(teacher_model, tvalloader, device, )

            logger.info(f"TargetVAL[{epo:>2d}]:    "
                        f"mIOU_car: {val_info['miou_car']:>7.4f}  "
                        f"TargetIOU: {np.array2string(val_info['iou'][1:].cpu().numpy(), precision=3, floatmode='fixed')}  "
                        f"mIOU: {val_info['miou']:>7.4f}  "
                        )
            mname = os.path.join(logdir, "model{}.pt".format(epo))
            print('saving', mname)
            torch.save(teacher_model.state_dict(), mname)


if __name__ == '__main__':
    version =  'v1.0-trainval'#'v1.0-mini'#
    grid_conf = {
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                     'Ncams': 6, 'up_scale': 4, 'H': 900, 'W': 1600,
                     'rand_resize': True, 'resize_lim': (0.20, 0.235), 'bot_pct_lim': (0.0, 0.22),
                     'rand_flip': True,
                     'rot': True, 'rot_lim': (-5.4, 5.4),
                     'color_jitter': True, 'color_jitter_conf': [0.2, 0.2, 0.2, 0.1],
                     'GaussianBlur': False, 'gaussion_c': (0, 2),
                     'final_dim': (128, 352),  # (224,480),#
                     'Aug_mode': 'hard',  # 'simple',#
                     'backbone': "efficientnet-b0",
                     }
    b = 10
    lr = 1e-3*b/4
    source_name_list = ['boston', 'singapore', 'day', 'dry']
    target_name_list = ['singapore', 'boston', 'night', 'rain']
    n = 0
    source = source_name_list[n]
    target = target_name_list[n]
    train(logdir='./ours_mapping_' + source+'_'+target+'_2', version=version, dataroot='',
          grid_conf=grid_conf, data_aug_conf=data_aug_conf,
          domain_gap=True, source=source, target=target, nsweeps=3,
          bsz=b, nworkers=6, lr=lr, weight_decay=1e-2, nepochs=24, )
