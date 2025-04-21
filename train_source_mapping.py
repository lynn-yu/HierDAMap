import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4,2,3,0,5,'
import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from data.nus_mapping import compile_data_mapping_source
from Our.LSS_model_da import LiftSplatShoot,LiftSplatShoot_mask_mix
from tools import *
from torch.optim import SGD, AdamW,Adam
from torch.optim.lr_scheduler import MultiStepLR,StepLR
from Our.confusion import BinaryConfusionMatrix,singleBinary
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
        # self.ce_criterion = nn.CrossEntropyLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

        # self.nll_criterion = nn.NLLLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

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
    model.eval()
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

    model.train()
    iou = total_intersect / (total_union+1e-7)
    miou = torch.mean(iou[1:])
    return {'iou': iou,'miou': miou,'miou_car': confusion_c.mean_iou}



def train(logdir,grid_conf,data_aug_conf,version, dataroot,nsweeps, domain_gap,source,target, bsz,nworkers,lr,weight_decay,nepochs,
          max_grad_norm=5.0,gpuid=0,):
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
    strainloader, tvalloader = compile_data_mapping_source(version, dataroot, data_aug_conf, grid_conf,nsweeps,  domain_gap,source,target,bsz,nworkers,flip=True)
    straingenerator = iter(strainloader)

    datalen = len(strainloader)

    device =  torch.device(f'cuda:{gpuid}')

    student_model = LiftSplatShoot_mask_mix(grid_conf, data_aug_conf, outC=4)
    student_model.to(device)
    student_model.train()




    epoch = int(nepochs)

    depth_weights = torch.Tensor([2.4, 1.2, 1.0, 1.1, 1.2, 1.4, 1.8, 2.3, 2.7, 3.5,
                                  3.6, 3.9, 4.8, 5.8, 5.4, 5.3, 5.0, 5.4, 5.3, 5.9,
                                  6.5, 7.0, 7.5, 7.5, 8.5, 10.3, 10.9, 9.8, 11.5, 13.1,
                                  15.1, 15.1, 16.3, 16.3, 17.8, 19.6, 21.8, 24.5, 24.5, 28.0,
                                  28.0]).to(device)
    loss_depth = DepthLoss(depth_weights).cuda(gpuid)
    loss_pv = Dice_Loss().cuda()
    loss_final = SegmentationLoss(class_weights=[1.0, 2.0, 2.0, 2.0], weight=1.0).cuda()
    loss_car = Dice_Loss().cuda()

    opt = AdamW(student_model.parameters(), lr=lr)
    sched = StepLR(opt, 10, 0.1)


    wp = 0.5
    print('weight pv:', wp)



    loss_nan_counter = 0
    np.random.seed()

    for epo in range(1,epoch):
        iteration = (epo-1)*datalen
        for i in range(datalen):
            ###source student
            try:
                imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s = next(straingenerator)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                straingenerator = iter(strainloader)
                imgs, rots, trans, intrins, post_rots, post_trans, lidars, binimgs_s, seg_mask_s = next(straingenerator)
            lidars_i = rearrange(lidars, 'b n c h w -> (b n) c h w')
            seg_mask = rearrange(seg_mask_s, 'b n c h w -> (b n) c h w')
            preds, depth, pv_out, x_bev, x_mini, mask_mini, pred_car = student_model(imgs.to(device), rots.to(device),
                                                                                     trans.to(device), intrins.to(device),
                                                                                     post_rots.to(device),
                                                                                     post_trans.to(device),
                                                                                     lidars_i.to(device), camdroup=False, )

            # plt.figure()
            # plt.imshow(binimgs_s['iou'][0][0])
            # plt.show()
            # plt.figure()
            # plt.imshow(binimgs_s['car'][0])
            # plt.show()
            binimgs = binimgs_s['index'].to(device)
            binimgs_car = binimgs_s['car'].to(device).unsqueeze(1)

            loss_f = loss_final(preds, binimgs)
            loss_f_car = loss_car(pred_car, binimgs_car)
            loss_p = wp * loss_pv(pv_out, seg_mask.cuda())
            loss_d, d_isnan = loss_depth(depth, lidars.to(device))
            loss_d = 0.1 * loss_d



            opt.zero_grad()
            loss = loss_f  + loss_p + loss_f_car
            if d_isnan:
                loss_nan_counter += 1
            else:
                loss = loss + loss_d
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            opt.step()

            if i > 0 and i % 10 == 0:
                _, _, iou = get_batch_iou_mapping(onehot_encoding_mapping(preds), binimgs_s['iou'].cuda())
                iou_car, miou_car = singleBinary(pred_car.sigmoid() > score_th,  binimgs_car > 0, dim=1)
                logger.info(f"EVAL[{int(epo):>3d}]: [{i:>4d}/{datalen}]:    "
                            f"lr {opt.param_groups[0]['lr']:>.2e}   "
                            f"loss: {loss.item():>7.4f}   "
                            f"loss_f: {loss_f.item():>7.4f}   "
                            f"loss_p: {loss_p.item():>7.4f}   "
                            f"mIOU_car: {miou_car:>7.4f}  "
                            f"IOU: {np.array2string(iou[1:].cpu().numpy(), precision=3, floatmode='fixed')}  "
                            )


            iteration += 1

        sched.step()
        if True:

            val_info = get_val_info(student_model, tvalloader, device, )
            logger.info(f"TargetVAL[{epo:>2d}]:    "
                         f"mIOU_car: {val_info['miou_car']:>7.4f}  "
                        f"TargetIOU: {np.array2string(val_info['iou'][1:].cpu().numpy(), precision=3, floatmode='fixed')}  "
                        f"mIOU: {val_info['miou']:>7.4f}  "

                        )
            mname = os.path.join(logdir, "model{}.pt".format(epo))
            print('saving', mname)
            torch.save(student_model.state_dict(), mname)




if __name__ == '__main__':
    version = 'v1.0-trainval'  #'v1.0-mini'#
    grid_conf = {
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                     'Ncams': 6, 'up_scale': 4, 'H': 900, 'W': 1600,
                     'rand_resize': True, 'resize_lim': (0.20, 0.235),'bot_pct_lim': (0.0, 0.22),
                     'rand_flip': True,
                     'rot': True, 'rot_lim': (-5.4, 5.4),
                     'color_jitter': True, 'color_jitter_conf': [0.2, 0.2, 0.2, 0.1],
                     'GaussianBlur': False, 'gaussion_c': (0, 2),
                     'final_dim': (128, 352),#(224,480),#
                     'Aug_mode': 'hard',  # 'simple',#
                     'backbone': "efficientnet-b0",
                     }
    b = 12
    source_name_list = ['boston', 'singapore', 'day', 'dry']
    target_name_list = ['singapore', 'boston', 'night', 'rain']
    n = 0
    source = source_name_list[n]
    target = target_name_list[n]
    train(logdir='./ours_source_mapping_' + source , version=version, dataroot='',
          grid_conf=grid_conf, data_aug_conf=data_aug_conf,
          domain_gap=True, source=source, target=target, nsweeps=3,
          bsz=b, nworkers=10, lr=3e-3, weight_decay=1e-2, nepochs=24, )