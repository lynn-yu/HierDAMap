import matplotlib.pyplot as plt
from cv2 import absdiff
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from data.tools import gen_dx_bx,QuickCumsum
import torch.nn.functional as F
from Our.erfnet import Encoder,Decoder
from einops import rearrange
import math
import numpy as np
from tools import *
from math import ceil
import random
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        b,c,h,w = x2.shape
        if x1.shape[2]!=h or x1.shape[3]!=w :
            x1 = F.interpolate(x1,(h,w), mode='bilinear', align_corners=True)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear',
                               align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        x1 = self.up4(x1)
        x2 = self.up2(x2)
        h,w = x3.shape[-2],x3.shape[-1]
        x1 = F.interpolate(x1,(h,w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, (h, w), mode='bilinear', align_corners=True)
        x3 = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x3)

class BevEncode(nn.Module):
    def __init__(self, inC, outC,aux_out=1):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )
        self.out_1 = nn.Conv2d(128, outC, kernel_size=1, padding=0)
        self.out_2 = nn.Conv2d(128, aux_out, kernel_size=1, padding=0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.up1(x3, x1)
        x = self.up2(x)
        x_new = self.out_1(x)
        x_car = self.out_2(x)

        return x_new, x_car

class CamEncode(nn.Module):
    def __init__(self, D, C):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.encoder_pv = Encoder(3, k=2)
        self.decoder_pv = Decoder(6, k=2)
        model_dict = self.encoder_pv.state_dict()
        path = '/data1/lsy/package/erfnet_pretrained.pth'
        modelCheckpoint = torch.load(path)
        new_dict = {}
        for k, v in modelCheckpoint.items():
            if 'encoder' in k:
                new_k = k[15:]
                new_dict[new_k] = v
        model_dict.update(new_dict)
        self.encoder_pv.load_state_dict(model_dict)
        print(
            'Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(modelCheckpoint),
                                                                                 len(new_dict)))
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up2 = Up2(320 + 112 + 40, 256)
        self.d2s = nn.Conv2d(256,128,kernel_size=1, padding=0)
        self.merge_s = nn.Conv2d(128,128,kernel_size=3, padding=1)
        self.s2d = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.merge_d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.depth_out = nn.Conv2d(256, self.D, kernel_size=1, padding=0)
        self.pv = nn.Conv2d(128, self.C, kernel_size=1, padding=0)
    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # Depth
        x_dep = self.get_eff_depth(x)#b 256 h w

        ##pv
        x_enc = self.encoder_pv(x)

        ##pv +depth
        x_dep_mid = self.d2s(x_dep)
        x_enc_final = self.merge_s(x_enc + x_dep_mid)
        pv_out = self.decoder_pv(x_enc_final)

        ###depth +pv
        x_enc_mid = self.s2d(x_enc)
        x_dep = self.merge_d(x_enc_mid + x_dep)
        depth = self.depth_out(x_dep)


        x_ = self.pv(x_enc_final)
        depth = self.get_depth_dist(depth)
        new_x = depth.unsqueeze(1) * x_.unsqueeze(2)

        # depth: [B*N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # new_x: [B*N, C, D, final_dim[0]//downsample, final_dim[1]//downsample]
        return depth, new_x, x_enc, pv_out

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        # print(x.shape)

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(x.shape)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x



        x = self.up2(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'])

        return x


    def forward(self, x, lidars):
        depth, new_x, x_mid,pv_out = self.get_depth_feat(x)

        return depth, new_x,x_mid,pv_out

class CamEncode_2(nn.Module):
    def __init__(self, D, C, upsample,aus=False,backbone="efficientnet-b0"):
        super(CamEncode_2, self).__init__()
        self.D = D
        self.C = C
        self.upsample = upsample

        self.trunk = EfficientNet.from_pretrained(backbone)

        self.decoder_pv = Decoder(6, k=2)
        if backbone == "efficientnet-b0":
            self.up2 = Up2(320 + 112 + 40, 512)
        elif backbone=="efficientnet-b4":
            self.up2 = Up2(448+160+56, 512)
        self.aux = aus
        if self.aux:

            self.depthnet = nn.Conv2d(512, self.D  + self.D  + self.C, kernel_size=1, padding=0)
        else:
            self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)
        self.conc = nn.Conv2d(self.C,128, kernel_size=1, padding=0)
    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, img):
        x, ft = self.get_eff_depth(img)
        # Depth


        x_ = self.depthnet(x)
        if self.aux:
            dep_p = x_[:, :self.D]#self.decoder_de(self.encoder_de(img))
            depth = self.get_depth_dist(x_[:, self.D:self.D+self.D])
            pv_fea = x_[:, self.D+self.D:(self.D + self.D + self.C)]
        else:
            dep_p = x_[:, :self.D]
            depth = self.get_depth_dist(x_[:, :self.D])
            pv_fea = x_[:, self.D:(self.D + self.C)]

        new_x = depth.unsqueeze(1) * pv_fea.unsqueeze(2)
        pv_out = self.decoder_pv(self.conc(pv_fea))

        # depth: [B*N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # new_x: [B*N, C, D, final_dim[0]//downsample, final_dim[1]//downsample]
        return depth, new_x, x, pv_out,dep_p

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        # print(x.shape)

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            # print(x.shape)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x



        x = self.up2(endpoints['reduction_5'], endpoints['reduction_4'], endpoints['reduction_3'])

        return x, 0

    def forward(self, x, lidars):
        depth, new_x, x_mid,pv_out,dep_p = self.get_depth_feat(x)

        return depth, new_x, x_mid,pv_out,dep_p

class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC,aux=False,decouple=False,aux_out=1):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        self.xbound = self.grid_conf['xbound']
        self.ybound = self.grid_conf['ybound']
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = dx.cuda()
        self.bx = bx.cuda()
        self.nx = nx.cuda()
        self.minidown = 8
        self.nx_2 = nx.cuda()
        self.nx_2[:2] = self.nx_2[:2] // self.minidown
        self.dx_2 = dx.cuda()
        self.dx_2[:2] = self.dx[:2] * self.minidown

        self.downsample = 32 // (int)(self.data_aug_conf['up_scale'])
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape

        if decouple:
            self.camencode = CamEncode(self.D, self.camC)
        else:
            self.camencode = CamEncode_2(self.D, self.camC, self.data_aug_conf['up_scale'], aus=aux,backbone=self.data_aug_conf['backbone'])

        self.bevencode = BevEncode(inC=self.camC, outC=outC,aux_out=aux_out)
        self.use_quickcumsum = True
        self.droup = nn.Dropout(0.5)
        self.bn = torch.nn.BatchNorm2d(64)
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x, lidars, mask=None):
        """Return B x N x D x H/downsample x W/downsample x C
        mask b*n c h w
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)
        # depth: [B*N, D, final_dim[0]//downsample, final_dim[1]//downsample]
        # new_x: [B*N, C, D, final_dim[0]//downsample, final_dim[1]//downsample]
        depth, x, x_mid, pv_out,dep_p = self.camencode(x, lidars)
        ###mask
        if mask is not None:
            lidars = rearrange(lidars, 'b n c h w-> (b n) c h w')
            mask = depth.unsqueeze(1) * mask.unsqueeze(2)
            mask = mask.view(B, N, 6, self.D, imH // self.downsample, imW // self.downsample)
            mask = mask.permute(0, 1, 3, 4, 5, 2)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, imH // self.downsample, imW // self.downsample)

        return depth, x, x_mid, pv_out, mask

    def voxel_pooling(self, geom_feats, x,is_av = False):
        # x: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample, C]
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)


        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        vi = geom_feats[0][0][0].detach().cpu().numpy()
        geom_feats = geom_feats.view(Nprime, 3)

        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)

        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans, lidars,cross_bev=None):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        depth, x, x_mid, pv_out,_ = self.get_cam_feats(x, lidars)

        # x: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample, C]
        x = self.voxel_pooling(geom, x)

        return depth, x, x_mid, pv_out





    def forward(self,  x, rots, trans, intrins, post_rots, post_trans, lidars,droup=False,camdroup=False,x_bev_p=None,cross_mix=False,cross_bev=None):


        depth, x_bev_init, x_mid, pv_out= self.get_voxels(x, rots, trans, intrins, post_rots, post_trans,lidars)

        if droup:
            b, c, h, w = x_bev_init.shape
            if x_bev_p is not None:
                x_bev_p = x_bev_p.detach()
                if np.random.choice([0,0, 1]):
                    x_bev_init = self.droup(x_bev_init)
                else:
                    if np.random.choice([0, 1]):
                        mask = torch.randint(0, 2, (b, c)).cuda()
                        mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
                        b1 = x_bev_init * mask
                        b2 = x_bev_p * (1 - mask)
                        x_bev_init = b1 + b2
                    else:
                        mask = torch.randint(0, 2, (b, h, w)).cuda()
                        mask = mask.unsqueeze(1).repeat(1, c, 1, 1)
                        b1 = x_bev_init * mask
                        b2 = x_bev_p * (1 - mask)
                        x_bev_init = b1 + b2

            else:
                x_bev_init = self.droup(x_bev_init)


        x, x_final = self.bevencode(x_bev_init)
        return x,depth,pv_out,x_bev_init

class LiftSplatShoot_mix(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf, outC,):
        super().__init__(grid_conf, data_aug_conf, outC,)

        self.ln = torch.nn.InstanceNorm2d(64)
        self.bn = torch.nn.BatchNorm2d(64)

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans, lidars,cross_bev=None):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        depth, x_voxel, x_mid, pv_out,_ = self.get_cam_feats(x, lidars)

        # x: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample, C]
        x = self.voxel_pooling(geom, x_voxel)

        return depth, x, x_voxel, pv_out


    def forward(self, x, rots, trans, intrins, post_rots, post_trans, lidars, droup=False, camdroup=False, x_bev_p=None,cross_img=None,cross_bev=None,num=6,mix_num=1,gt=None,pl=None):
        img_t =x
        depth, x_bev_init, x_voxel, pv_out = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans, lidars)
        x_bev_init_ori = x_bev_init
        if droup:
            b, c, h, w = x_bev_init.shape
            if x_bev_p is not None:
                x_bev_p = x_bev_p.detach()
                if np.random.choice([0, 1]):
                    x_bev_init = self.droup(x_bev_init)
                else:
                    if np.random.choice([0, 1]):
                            mask = torch.randint(0, 2, (b, c)).cuda()
                            mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
                            b1 = x_bev_init * mask
                            b2 = x_bev_p * (1 - mask)
                            x_bev_init = b1 + b2
                    else:
                            mask = torch.randint(0, 2, (b, h, w)).cuda()
                            mask = mask.unsqueeze(1).repeat(1, c, 1, 1)
                            b1 = x_bev_init * mask
                            b2 = x_bev_p * (1 - mask)
                            x_bev_init = b1 + b2
            else:
                x_bev_init = self.droup(x_bev_init)


        x, x_final = self.bevencode(x_bev_init)



        if cross_bev is not None:
            b, c, h, w = cross_bev.shape
            cross_bev = cross_bev.detach()

            #vier mask
            mask_lap = get_view_mask()#6 200 200
            mask_nonlap = get_view_mask_nonlap()  # 6 200 200
            mb = torch.randint(0,num,(b,)).cuda()
            mask_lap = get_mask(mask_lap,mb)
            mask_nonlap = get_mask(mask_nonlap,mb)
            mask_t = torch.ones_like(mask_lap)-mask_lap
            mask_union = mask_lap - mask_nonlap
            #x_bev_mix = mask_nonlap * cross_bev + mask_t * x_bev_init_ori + 0.5 * mask_union*(cross_bev+x_bev_init_ori)

            x_bev_mix = mask_t * x_bev_init_ori

            x_mix,_ = self.bevencode(x_bev_mix.float())
            #self.show(mask_b[0][0])
            # self.show_fea(cross_bev[0])
            # self.show_fea(x_bev_init[0])
            # self.show_fea(x_bev_mix[0])

        else:

            x_mix = x_bev_init
            mb = 0

        return x, depth, pv_out, x_bev_init,x_mix,mb


from .loss_edl import edl_mse_loss,edl_log_loss,edl_digamma_loss


class LiftSplatShoot_mask_mix(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf, outC,aux_out=1,mid_ch=1):
        super().__init__(grid_conf, data_aug_conf, outC, aux_out=aux_out)
        self.ln = torch.nn.InstanceNorm2d(64)
        self.bn = torch.nn.BatchNorm2d(64)
        self.mid_out = nn.Sequential(

            nn.Conv2d(self.camC, self.camC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.camC),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.camC, mid_ch, kernel_size=1, padding=0),
        )  # nn.Conv2d(self.camC, 6, kernel_size=1, padding=0, stride=1)
        self.car_out = nn.Sequential(
            nn.Conv2d(self.camC, self.camC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.camC),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.camC, mid_ch, kernel_size=1, padding=0),
        )

        self.mid_ch = mid_ch

    def show(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans,post_trans_s=None,post_rots_s=None,mb=None,x=None,sx=None,seg=None,seg_car=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        if mb:
            b,n,c,h,w = seg_car.shape
            h_d = self.data_aug_conf['final_dim'][0]//self.downsample
            w_d = self.data_aug_conf['final_dim'][1]//self.downsample
            post_trans = post_trans.view(B, N, 1, 1, 1, 3).repeat(1, 1, 41, h_d,w_d, 1)
            post_rots = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).repeat(1, 1, 41,  h_d,w_d, 1,1)
            post_trans_s = post_trans_s.view(B, N, 1, 1, 1, 3).repeat(1, 1, 41,  h_d,w_d, 1)
            post_rots_s = torch.inverse(post_rots_s).view(B, N, 1, 1, 1, 3, 3).repeat(1, 1, 41,  h_d,w_d, 1, 1)
            seg_car = seg_car.repeat(1,1,41,1,1)
            seg_car_inv = torch.ones_like(seg_car)-seg_car

            post_trans = post_trans*seg_car_inv.unsqueeze(-1) + seg_car.unsqueeze(-1) * post_trans_s
            post_rots = post_rots * seg_car_inv.unsqueeze(-1).unsqueeze(-1) +seg_car.unsqueeze(-1).unsqueeze(-1) * post_rots_s

            points = self.frustum - post_trans
            points = torch.inverse(post_rots).matmul(points.unsqueeze(-1))

        else:
            # undo post-transformation
            # B x N x D x H x W x 3
            points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
            points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points,x,seg


    def get_dep_onehot(self, prob):
        b, d, h, w = prob.shape
        depth = torch.argmax(prob, dim=1)  # b h w
        pred_pp = torch.zeros_like(prob)  # b d h w
        bb = torch.linspace(0, b - 1, b).cuda()
        bb = bb.unsqueeze(-1).unsqueeze(-1).repeat(1, h, w)
        x = torch.linspace(0, h - 1, h)
        y = torch.linspace(0, w - 1, w)
        x, y = torch.meshgrid(x, y, indexing='ij')
        pred_pp[bb.long(), depth.long(), x[:].long(), y[:].long()] = 1
        return pred_pp

    def get_cam_feats(self, x, lidars, mask=None, epoch=0):
        """Return B x N x D x H/downsample x W/downsample x C
        mask b*n c h w
        """
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)

        depth, x, x_mid, pv_out, dep_p = self.camencode(x, lidars)

        ###mask
        if mask is not None:
            depth_un = self.get_dep_onehot(depth)
            loss_un = 0
            mask_p = depth_un.unsqueeze(1) * mask.unsqueeze(2)
            mask_p = mask_p.view(B, N, self.mid_ch, self.D, imH // self.downsample, imW // self.downsample)
            mask_p = mask_p.permute(0, 1, 3, 4, 5, 2)
        else:
            loss_un = 0
            mask_p = None

        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, imH // self.downsample, imW // self.downsample)

        return depth, x, x_mid, pv_out, mask_p, mask_p

    def voxel_pooling_2(self, geom_feats, x,is_av = False):
        # x: [B, N, D, final_dim[0]//downsample, final_dim[1]//downsample, C]
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx_2).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx_2[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx_2[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx_2[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx_2[1] * self.nx_2[2] * B) \
                + geom_feats[:, 1] * (self.nx_2[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx_2[2], self.nx_2[0], self.nx_2[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels_2(self, x, rots, trans, intrins, post_rots, post_trans, lidars, seg_mask,post_trans_s=None,post_rots_s=None,mb=None,sx=None,seg_mask_car=None,is_av = False):
        geom,x,seg_mask_o = self.get_geometry(rots, trans, intrins, post_rots, post_trans,post_trans_s,post_rots_s,mb,x,sx,seg_mask,seg_mask_car)

        depth, x, x_mid, pv_out, new_mask, gt_mask = self.get_cam_feats(x, lidars, seg_mask_o)
        x = self.voxel_pooling(geom, x,is_av)
        if seg_mask is not None:
            seg_mask = self.voxel_pooling_2(geom, new_mask,is_av)

        return depth, x, x_mid, pv_out, seg_mask,seg_mask_o

    def get_mid_out_loss(self, x_mid, mask, epoch=0):
        mask = 1.0 * (mask > 0.1)
        _, _, h, w = x_mid.shape
        loss = self.loss_d(x_mid,
                           mask)  # +0.5*self.get_edl_loss(x_mid,mask,epoch)#self.loss_d(x_mid,mask)#((x_mid.sigmoid()-mask)).mean()#
        return loss

    def get_mid_out(self, x):
        _, _, h, w = x.shape

        x = F.interpolate(x, size=(h // self.minidown, w // self.minidown), mode='bilinear', align_corners=True)
        x_mid = self.mid_out(x)
        return x_mid

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, lidars, droup=False, camdroup=False, x_bev_p=None,is_av=False,
                cross_mix=False, cross_bev=None, seg_mask=None, epo=0,source_img=None,source_trans=None,source_rots=None,seg_mask_car=None):
        b,n,c,h,w = x.shape
        if seg_mask is not None:
            if self.mid_ch ==1:
                seg_mask = seg_mask[:, 2].unsqueeze(1)
            else:
                seg_mask = seg_mask[:, 1:3]
            _, _, h, w = seg_mask.shape
            seg_mask = F.interpolate(seg_mask, size=(h // self.downsample, w // self.downsample), mode='bilinear',
                                     align_corners=True)
        if seg_mask_car is not None:
                _, _, h, w = seg_mask_car.shape
                seg_mask_car = F.interpolate(seg_mask_car, size=(h // self.downsample, w // self.downsample), mode='bilinear',
                                     align_corners=True)
                seg_mask_car = rearrange(seg_mask_car,'(b n) d h w -> b n d h w',b=b,n=n)




        depth, x_bev_init, x_mid, pv_out, mask_b,seg_mask_change = self.get_voxels_2(x, rots, trans, intrins, post_rots, post_trans,
                                                                     lidars, seg_mask,post_trans_s=source_trans,post_rots_s=source_rots,
                                                                                     mb=cross_mix,sx=source_img,seg_mask_car=seg_mask_car,
                                                                                     is_av=is_av,
                                                                                     )

        x_mini = self.get_mid_out(x_bev_init)
        #x_car = self.car_out(x_bev_init)
        if mask_b is not None:
            mask_b = 1.0 * (mask_b > 0.5)
        if droup:
            b, c, h, w = x_bev_init.shape
            if x_bev_p is not None:
                x_bev_p = x_bev_p.detach()
                if np.random.choice([0, 0, 1]):
                    x_bev_init = self.droup(x_bev_init)
                else:
                    if np.random.choice([0, 1]):
                        mask = torch.randint(0, 2, (b, c)).cuda()
                        mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
                        b1 = x_bev_init * mask
                        b2 = x_bev_p * (1 - mask)
                        x_bev_init = b1 + b2
                    else:
                        mask = torch.randint(0, 2, (b, h, w)).cuda()
                        mask = mask.unsqueeze(1).repeat(1, c, 1, 1)
                        b1 = x_bev_init * mask
                        b2 = x_bev_p * (1 - mask)
                        x_bev_init = b1 + b2
            else:
                x_bev_init = self.droup(x_bev_init)

        x, x_car = self.bevencode(x_bev_init)
        return x, depth, pv_out, x_bev_init, x_mini, mask_b,x_car




