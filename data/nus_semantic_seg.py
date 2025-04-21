import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode
from collections import OrderedDict
import torch
from shapely.strtree import STRtree
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.utils.data_classes import LidarPointCloud
from data.splits import create_splits_scenes
from data.tools import *
from nuscenes.utils.data_classes import Box
import cv2
from depth_anything.dpt import DPT_DINOv2
from Our.depth_pre import pre_depth_img
from data.predict_mask import Predictor
from san.data.datasets.register_coco_stuff_164k import COCO_CATEGORIES
IMG_ORIGIN_H = 900
IMG_ORIGIN_W = 1600
camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
]
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
}
def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage",
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps
class SemanticNuscData(torch.utils.data.Dataset):
    def __init__(self, version,dataroot, data_aug_conf, grid_conf, nsweeps, is_train, domain_gap,domain, domain_type):
        self.nusc = NuScenes(version=version,
                    dataroot=dataroot,
                    verbose=False)
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        ###domain
        self.domain_gap = domain_gap
        self.domain = domain
        self.domain_type = domain_type
        ### lidar
        self.downsample = 32 // (int)(self.data_aug_conf['up_scale'])
        self.nsweeps = nsweeps

        self.nusc_maps = get_nusc_maps(self.nusc.dataroot)
        self.scene2map = {}
        for rec in self.nusc.scene:
            log = self.nusc.get('log', rec['log_token'])
            self.scene2map[rec['name']] = log['location']
        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        h = grid_conf['xbound'][1] - grid_conf['xbound'][1]
        w = grid_conf['ybound'][1] - grid_conf['ybound'][1]
        self.patch = (w,h)
        self.aug_mode = data_aug_conf['Aug_mode']

        print(self)
        print(self.__len__(),data_aug_conf['Aug_mode'])

    def load_map_data(self,dataroot, location):

        # Load the NuScenes map object
        nusc_map = NuScenesMap(dataroot, location)

        map_data = OrderedDict()
        for layer in STATIC_CLASSES:

            # Retrieve all data associated with the current layer
            records = getattr(nusc_map, layer)
            polygons = list()

            # Drivable area records can contain multiple polygons
            if layer == 'drivable_area':
                for record in records:

                    # Convert each entry in the record into a shapely object
                    for token in record['polygon_tokens']:
                        poly = nusc_map.extract_polygon(token)
                        if poly.is_valid:
                            polygons.append(poly)
            else:
                for record in records:

                    # Convert each entry in the record into a shapely object
                    poly = nusc_map.extract_polygon(record['polygon_token'])
                    if poly.is_valid:
                        polygons.append(poly)

            # Store as an R-Tree for fast intersection queries
            map_data[layer] = STRtree(polygons)

        return map_data

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (
                        rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        if self.domain_gap:
            split = {
                'boston': {'strain': 'boston', 'ttrain': 'boston_train', 'tval': 'boston_val'},
                'singapore': {'strain': 'singapore', 'ttrain': 'singapore_train', 'tval': 'singapore_val'},
                'singapore_day': {'strain': 'singapore_day', 'ttrain': 'singapore_day_train',
                                  'tval': 'singapore_day_val'},
                'singapore_day_train': {'strain': 'singapore_day_train', 'ttrain': None, 'tval': None},
                'day': {'strain': 'day', 'ttrain': None, 'tval': None},
                'night': {'strain': None, 'ttrain': 'night_train', 'tval': 'night_val'},
                'dry': {'strain': 'dry', 'ttrain': None, 'tval': None},
                'rain': {'strain': None, 'ttrain': 'rain_train', 'tval': 'rain_val'},
                'nuscenes': {'strain': 'train', 'ttrain': 'train', 'tval': 'val'},
                'lyft': {'strain': 'lyft_train', 'ttrain': 'lyft_train', 'tval': 'lyft_val'},
            }[self.domain][self.domain_type]
        else:
            split = {
                'v1.0-trainval': {True: 'train', False: 'val'},
                'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
            }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def sample_augmentation_simple_strong(self):
        fH, fW =  self.data_aug_conf['final_dim']#128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)#（1，1）变换尺度
        resize_dims = (fW, fH)#（128,352）目标数据尺寸
        if self.is_train:
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]) :
                colorjt = True
            else:
                colorjt = False
        else:
            colorjt = False
        return resize, resize_dims,colorjt
    def sample_augmentation_simple(self):
        fH, fW =  self.data_aug_conf['final_dim']#128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)#（1，1）变换尺度
        resize_dims = (fW, fH)#（128,352）目标数据尺寸
        if self.is_train:
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]) :
                colorjt = True
            else:
                colorjt = False
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            colorjt = False
            rotate= 0

        return resize, resize_dims,colorjt,rotate

    def sample_augmentation_weak(self):

        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if self.data_aug_conf['rand_resize']:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                resize = max(fH/H, fW/W)
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)  # （1，1）变换尺度
            # resize_dims = (fW, fH)  # （128,352）目标数据尺寸
            # newW, newH = resize_dims
            # crop_h = int((1 - 0)*newH) - fH
            # crop_w = int(max(0, newW - fW) / 2)
            # crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate =  np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop,flip, rotate

    def sample_augmentation(self):

        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if self.data_aug_conf['rand_resize']:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                resize = max(fH/H, fW/W)
                resize_dims = (int(W*resize), int(H*resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True

            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
            if self.data_aug_conf['color_jitter']:
                colorjt = True
            else:
                colorjt = False
            if self.data_aug_conf['GaussianBlur'] and np.random.choice([0, 1]):
                gauss = np.random.uniform(*self.data_aug_conf['gaussion_c'])
            else:
                gauss = 0
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - 0) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            colorjt = False
            gauss = 0
        return resize, resize_dims, crop, flip, rotate, colorjt,gauss

    def get_label(self,mask):
        mask = torch.tensor(np.array(mask))
        vi = np.array(mask)
        mask_per = mask == 1
        mask_car = torch.logical_and(mask < 10, mask > 1)
        mask_tra = mask == 10
        mask_road = torch.logical_or(mask == 138, mask == 136)

        mask_build = torch.logical_or(mask == 147, mask == 85)

        mask_all = torch.stack([mask_road,mask_car,mask_build,mask_tra,mask_per])#,mask_build,mask_tree
        mask_all = torch.cat([(~torch.any(mask_all, axis=0)).unsqueeze(0), mask_all])


        return mask_all*1.0

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        dep_image=[]
        seg_mask = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])

            path = 'xx/nuscenes_semanti_mask'
            cam_path = os.path.join(path, samp['filename'][8:])
            image_gray = Image.open(cam_path)


            imgname = os.path.join(self.nusc.dataroot, samp['filename'])

            img = Image.open(imgname)


            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])
            #print(cam,tran)
            # augmentation (resize, crop, horizontal flip, rotate)
            if True:
                resize, resize_dims, crop, flip, rotate, colorjt,gauss = self.sample_augmentation()
                img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,False,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate,
                                                           colorjt=colorjt,
                                                           colorjt_conf=self.data_aug_conf['color_jitter_conf']
                                                           )
                image_gray = np.array(image_gray) + 1
                image_gray = Image.fromarray(image_gray.astype(np.uint8))
                mask, _,_ = img_transform(image_gray, post_rot, post_tran,False,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate,
                                                           colorjt=False,
                                                           colorjt_conf=self.data_aug_conf['color_jitter_conf'])






            mask = self.get_label(mask)
            if self.data_aug_conf['Ncams']==6:
                seg_mask.append(mask)
                dep_image.append(normalize_img(img))
                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                imgs.append(normalize_img(img))
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)


        return (torch.stack(dep_image),torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans),torch.stack(seg_mask))

    def get_binimg(self, rec,mask):
        img = np.zeros((self.nx[0], self.nx[1]))

        egopose = self.nusc.get('ego_pose',
                                    self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])#ego2global_translation
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        for tok in rec['anns']:
                inst = self.nusc.get('sample_annotation', tok)

                if not inst['category_name'].split('.')[0] == 'vehicle':
                        continue
                box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T
                pts = np.round(
                    (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                    ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)
        img = torch.Tensor(img)


        return img

    def get_divider(self,rec,mask):
        img = np.zeros((self.nx[0], self.nx[1]))
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        poly_names = []
        line_names = ['road_divider', 'lane_divider']
        lmap = get_local_map(self.nusc_maps[map_name], center,
                             50.0, poly_names, line_names)
        for name in poly_names:
            for la in lmap[name]:
                pts = np.round((la - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)
        for name in line_names:
            for la in lmap[name]:
                pts = np.round((la - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                # valid_pts = np.logical_and((pts[:, 0] < 200), np.logical_and((pts[:, 0] >= 0),
                #             np.logical_and((pts[:, 1] >= 0), (pts[:, 1] < 200))))
                # img[pts[valid_pts, 0], pts[valid_pts, 1]] = 1.0
                # cv2.fillPoly(img, [pts], 1.0)
                cv2.polylines(img, [pts], isClosed=False, color=1.0, thickness=2)
        img = torch.Tensor(img)
        # img = torch.flip(img,dims=[1])
        mask.append(img)
        return mask

    def get_static(self,rec):

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        poly_names = ['road_segment','lane', 'ped_crossing', 'walkway', 'stop_line','carpark_area']#
        line_names = []
        lmap = get_local_map(self.nusc_maps[map_name], center,
                             50, poly_names, line_names)
        mask=[]

        img = np.zeros((self.nx[0], self.nx[1]))
        for name in ['road_segment','lane']:
            for la in lmap[name]:
                pts = np.round((la - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)
        mask.append(torch.Tensor(img))

        for name in ['ped_crossing', 'walkway', 'stop_line','carpark_area']:
            img = np.zeros((self.nx[0], self.nx[1]))
            for la in lmap[name]:
                pts = np.round((la - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)
            mask.append(torch.Tensor(img))


        return mask


    def get_lidar_data_to_img(self, rec, post_rots, post_trans, cams, downsample, nsweeps):
        points_imgs,_,_ = get_lidar_data_to_img(self.nusc, rec, data_aug_conf=self.data_aug_conf,
                                            grid_conf=self.grid_conf, post_rots=post_rots, post_trans=post_trans,
                                            cams=cams, downsample=downsample, nsweeps=nsweeps, min_distance=2.2,)
        return torch.Tensor(points_imgs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        depimg,imgs, rots, trans, intrins, post_rots, post_trans,seg_mask = self.get_image_data(rec, cams)
        lidar = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)

        mask = self.get_static(rec)
        gt = self.get_divider(rec,mask)
        gt = torch.stack(gt)
        #gt = torch.cat([(~torch.any(gt, axis=0)).unsqueeze(0), gt])
        car = self.get_binimg(rec,None)
        gt = torch.cat([gt,car.unsqueeze(0)],dim=0)
        return imgs, rots, trans, intrins, post_rots, post_trans,lidar,gt,seg_mask

    def __len__(self):

        return len(self.ixes)

class FlipSemanticNuscData(SemanticNuscData):
    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        dep_image=[]
        post_rots_flip = []
        post_trans_flip = []
        imgs_flip = []
        seg_mask = []
        seg_mask_flip = []
        seg_mask_3 = []
        post_rots_3 = []
        post_trans_3 = []
        imgs_3 = []
        flip_g = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            path = 'xx/nuscenes_semanti_mask'
            cam_path = os.path.join(path, samp['filename'][8:])
            mask_o = Image.open(cam_path)


            imgname = os.path.join(self.nusc.dataroot, samp['filename'])

            img_o = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            if self.aug_mode == 'hard':
                mask_o = np.array(mask_o) + 1
                mask_o = Image.fromarray(mask_o.astype(np.uint8))
                resize, resize_dims, crop, flip, rotate, colorjt,guass = self.sample_augmentation()
                img, post_rot2, post_tran2 = img_transform(img_o, post_rot, post_tran,guass,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate,
                                                           colorjt=colorjt,
                                                           colorjt_conf=self.data_aug_conf['color_jitter_conf']
                                                           )
                mask,_,_ = img_transform(mask_o, post_rot, post_tran,0,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate,
                                                           colorjt=False,
                                                           colorjt_conf=self.data_aug_conf['color_jitter_conf']
                                                           )
                mask = self.get_label(mask)
                seg_mask.append(mask)
                img_3, post_rot3, post_tran3 = img_transform(img_o, post_rot, post_tran, guass,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=0,
                                                           colorjt=colorjt,
                                                           colorjt_conf=self.data_aug_conf['color_jitter_conf']
                                                           )
                mask_3, _, _ = img_transform(mask_o, post_rot, post_tran, 0,
                                           resize=resize,
                                           resize_dims=resize_dims,
                                           crop=crop,
                                           flip=flip,
                                           rotate=0,
                                           colorjt=False,
                                           colorjt_conf=self.data_aug_conf['color_jitter_conf']
                                           )
                mask_3 = self.get_label(mask_3)
                seg_mask_3.append(mask_3)
                resize, resize_dims,_,_ = self.sample_augmentation_simple()
                img_flip, post_rot2_flip, post_tran2_flip = img_transform_simple(img_o, resize, resize_dims)
                mask_o, _, _ = img_transform_simple(mask_o, resize, resize_dims)
                mask_o = self.get_label(mask_o)
                seg_mask_flip.append(mask_o)
            else:
                resize, resize_dims, flip, rotate = self.sample_augmentation_simple()
                if self.is_train and self.data_aug_conf['color_jitter']:
                    colorjt = True
                else:
                    colorjt = False
                img, post_rot2, post_tran2 = img_transform_strong(img_o, resize, resize_dims, flip=flip,
                                                                      colorjt=colorjt,colorjt_conf=self.data_aug_conf['color_jitter_conf'],
                                                                    rotate=rotate,gauss=False)
                mask_o = np.array(mask_o) + 1
                mask_o = Image.fromarray(mask_o.astype(np.uint8))
                mask, post_rot2, post_tran2 = img_transform_strong(mask_o, resize, resize_dims, flip=flip,
                                                                    colorjt=False,colorjt_conf=self.data_aug_conf['color_jitter_conf'],
                                                                    rotate=rotate, gauss=False)
                mask = self.get_label(mask)
                seg_mask.append(mask)
                img_flip, post_rot2_flip, post_tran2_flip= img_transform_strong(img_o, resize, resize_dims, flip=False,colorjt=False,rotate=0,gauss=False)
                mask_o,_,_ =  img_transform_strong(mask_o, resize, resize_dims, flip=False,colorjt=False,rotate=0,gauss=False)

            flip_g.append(1*flip)


            dep_image.append(normalize_img(img))
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            post_tran_f = torch.zeros(3)
            post_rot_f = torch.eye(3)
            post_tran_f[:2] = post_tran2_flip
            post_rot_f[:2, :2] = post_rot2_flip

            post_tran_3 = torch.zeros(3)
            post_rot_3 = torch.eye(3)
            post_tran_3[:2] = post_tran3
            post_rot_3[:2, :2] = post_rot3

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            imgs_flip.append(normalize_img(img_flip))
            post_trans_flip.append(post_tran_f)
            post_rots_flip.append(post_rot_f)

            imgs_3.append(normalize_img(img_3))
            post_trans_3.append(post_tran_3)
            post_rots_3.append(post_rot_3)

        return (torch.stack(dep_image),torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans),torch.stack(imgs_flip),torch.stack(post_trans_flip),torch.stack(post_rots_flip),
                torch.stack(seg_mask),torch.stack(seg_mask_flip),
                torch.stack(imgs_3), torch.stack(post_trans_3), torch.stack(post_rots_3),torch.stack(seg_mask_3),
                )

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        depimg,imgs, rots, trans, intrins, post_rots, post_trans,imgs_flip,post_trans_flip,post_rots_flip,seg_mask,seg_mask_o,\
            img_3,post_trans_3,post_rots_3,seg_mask_3= self.get_image_data(rec, cams)
        lidar = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)

        mask = self.get_static(rec)
        gt = self.get_divider(rec, mask)
        gt = torch.stack(gt)
        #gt = torch.cat([(~torch.any(gt, axis=0)).unsqueeze(0), gt])
        car = self.get_binimg(rec, None)
        gt = torch.cat([gt,car.unsqueeze(0)], dim=0)
        return imgs, rots, trans, intrins, post_rots, post_trans,lidar,gt,imgs_flip,post_rots_flip,post_trans_flip,seg_mask,seg_mask_o#,\img_3,post_rots_3,post_trans_3,seg_mask_3


def worker_rnd_init(x):
    np.random.seed(13 + x)

def compile_data(version, dataroot, data_aug_conf, grid_conf,nsweeps,  domain_gap,source,target,  bsz,nworkers):


    traindata = SemanticNuscData(version,dataroot,  data_aug_conf, grid_conf, nsweeps=nsweeps,  domain_gap=domain_gap,is_train=True,domain=source, domain_type='strain')
    valdata = SemanticNuscData(version,dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps,  domain_gap=domain_gap,is_train=False,domain=target, domain_type='tval')

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader

def compile_data_domain(version, dataroot, data_aug_conf, grid_conf,nsweeps,  domain_gap,source,target,  bsz,nworkers,flip=False):


    straindata = SemanticNuscData(version,dataroot,  data_aug_conf, grid_conf, nsweeps=nsweeps,  domain_gap=domain_gap,is_train=True,domain=source, domain_type='strain')
    if flip:
        ttraindata = FlipSemanticNuscData(version, dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps,
                                      domain_gap=domain_gap, is_train=True, domain=target, domain_type='ttrain')
    else:
        ttraindata = SemanticNuscData(version, dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps, domain_gap=domain_gap,is_train=True, domain=target, domain_type='ttrain')

    valdata = SemanticNuscData(version,dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps,  domain_gap=domain_gap,is_train=False,domain=target, domain_type='tval')

    strainloader = torch.utils.data.DataLoader(straindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    ttrainloade = torch.utils.data.DataLoader(ttraindata, batch_size=bsz,
                                               shuffle=True,
                                               num_workers=nworkers,
                                               drop_last=True,
                                               worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return strainloader, ttrainloade,valloader



def compile_data_test(version, dataroot, data_aug_conf, grid_conf,nsweeps,  domain_gap,source,target,  bsz,nworkers,domain_type='tval',is_train=False):




    valdata = SemanticNuscData(version,dataroot, data_aug_conf, grid_conf, nsweeps=nsweeps,  domain_gap=domain_gap,is_train=is_train,domain=target, domain_type=domain_type)


    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return valloader

def vis_img(image):
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
    plt.show()
def vis_mask(image):

    imgs = [image[i][1] for i in range(image.shape[0])]
    a = np.hstack(imgs[:3])
    b = np.hstack(imgs[3:])
    vi = np.vstack((a, b))
    plt.figure()
    plt.axis('off')
    plt.imshow(vi)
    plt.show()
def visual(masks,classes=['road_segment','ped_crossing', 'walkway', 'stop_line','carpark_area','divider'],background=(110,110,110)):

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    plt.figure()
    plt.axis('off')
    plt.imshow(canvas)
    plt.show()

from einops import rearrange
if __name__ == '__main__':

    grid_conf = {
        'xbound': [-51.2, 51.2, 0.10],
        'ybound': [-51.2, 51.2, 0.10],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 0.5],
    }
    data_aug_conf = {'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 6,'H':900,'W':1600,
                     'up_scale': 4,'rand_resize': False,'resize_lim': (0.20, 0.235),
                     'rand_flip': False,'rot_lim':(-5.4, 5.4),'bot_pct_lim': (0.0, 0.22),
                     'color_jitter': False, 'color_jitter_conf': [0.2, 0.2, 0.2, 0.1],
                     'GaussianBlur': False, 'gaussion_c': (0, 2),
                      'final_dim': (128, 352),'Aug_mode':'hard',#'simple',#
                     }
    source_name_list = ['boston', 'singapore', 'day', 'dry']
    target_name_list = ['singapore', 'boston', 'night', 'rain']
    n = 0
    source_name=source_name_list[n]
    target_name=target_name_list[n]
    data = SemanticNuscData(version='v1.0-mini',#'v1.0-trainval',
                    dataroot='',is_train=False,data_aug_conf=data_aug_conf,grid_conf=grid_conf,nsweeps=4,
                            domain_gap=True,domain=source_name, domain_type='strain',)
