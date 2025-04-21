import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset
from glob import glob
from .tools import (get_lidar_data, get_lidar_data_to_img, img_transform,
                    normalize_img, gen_dx_bx, get_local_map, get_nusc_maps)
from .tools import *
from .tools import img_transform_simple
from .vector_map import VectorizedLocalMap
from .rasterize import preprocess_map
from .splits import create_splits_scenes
IMG_ORIGIN_H = 900
IMG_ORIGIN_W = 1600

class MappingNuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, nsweeps, domain_gap,
                 domain, domain_type, is_lyft=False):
        self.nusc = nusc
        self.is_lyft = is_lyft
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.domain_gap = domain_gap
        self.domain = domain
        self.downsample = 32 // (int)(self.data_aug_conf['up_scale'])

        self.nsweeps = nsweeps
        if self.is_lyft is False:
            self.nusc_maps = get_nusc_maps(self.nusc.dataroot)
        self.domain_type = domain_type

        self.scene2map = {}
        for rec in self.nusc.scene:
            log = self.nusc.get('log', rec['log_token'])
            self.scene2map[rec['name']] = log['location']

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        if self.is_lyft is False:
            self.fix_nuscenes_formatting()
        self.thickness = 3
        self.angle_class = 36
        self.xbound = self.grid_conf['xbound']
        self.ybound = self.grid_conf['ybound']
        x = self.xbound[1] - self.xbound[0]  # [-30.0, 30.0, 0.15],
        y = self.ybound[1] - self.ybound[0]  # [-15.0, 15.0, 0.15],
        self.patch_size = (x, y)  # 60 30
        canvasx = (self.xbound[1] - self.xbound[0]) / self.xbound[2]  # 400
        canvasy = (self.ybound[1] - self.ybound[0]) / self.ybound[2]  # 200
        self.canvas_size = (int(canvasx), int(canvasy))  # 512 512
        self.vector_map = VectorizedLocalMap('/data0/lsy/dataset/nuScenes/v1.0', patch_size=self.patch_size,
                                             canvas_size=self.canvas_size)
        self.aug_mode = data_aug_conf['Aug_mode']
        self.use_lidar = data_aug_conf['lidar'] if 'lidar' in data_aug_conf else True

        print(self)
        print(self.__len__(), data_aug_conf['Aug_mode'])
        print(self.use_lidar)

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
    def __len__(self):
        return len(self.ixes)

    def sample_augmentation_simple(self):
        fH, fW = self.data_aug_conf['final_dim']  # 128 352
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)  # （1，1）变换尺度
        resize_dims = (fW, fH)  # （128,352）目标数据尺寸
        if self.is_train:
            if self.data_aug_conf['color_jitter']:
                colorjt = True
            else:
                colorjt = False
            rotate = 0
        else:
            colorjt = False
            rotate = 0

        return resize, resize_dims, colorjt, rotate

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
            if self.data_aug_conf['rot']:
                rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
            else:
                rotate = 0
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

    def get_label(self, mask):
        mask = torch.tensor(np.array(mask))
        vi = np.array(mask)
        mask_per = mask == 1
        mask_car = torch.logical_and(mask < 10, mask > 1)
        mask_tra = mask == 10
        mask_road = torch.logical_or(mask == 138, mask == 136)

        mask_sky = mask == 146
        mask_build = torch.logical_or(mask == 147, mask == 85)
        mask_tree = mask == 158
        mask_all = torch.stack([mask_road, mask_car, mask_build, mask_tra, mask_per])  # ,mask_build,mask_tree
        mask_all = torch.cat([(~torch.any(mask_all, axis=0)).unsqueeze(0), mask_all])


        return mask_all * 1.0

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        dep_image = []
        seg_mask = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])

            path = '/data1/lsy/nuscenes_semanti_mask'
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
            # print(cam,tran)
            # augmentation (resize, crop, horizontal flip, rotate)
            if self.aug_mode == 'hard':
                resize, resize_dims, crop, flip, rotate, colorjt, gauss = self.sample_augmentation()
                img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran, False,
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
                mask, _, _ = img_transform(image_gray, post_rot, post_tran, False,
                                           resize=resize,
                                           resize_dims=resize_dims,
                                           crop=crop,
                                           flip=flip,
                                           rotate=rotate,
                                           colorjt=False,
                                           colorjt_conf=self.data_aug_conf['color_jitter_conf'])



            else:
                resize, resize_dims, colorjt , rotate = self.sample_augmentation_simple()


                img, post_rot2, post_tran2 = img_transform_weak(img, resize, resize_dims,
                                                                  colorjt=colorjt,
                                                                  colorjt_conf=self.data_aug_conf['color_jitter_conf'],)

                image_gray = np.array(image_gray) + 1
                image_gray = Image.fromarray(image_gray.astype(np.uint8))
                mask, _,_ = img_transform_weak(image_gray, resize, resize_dims,
                                                                  colorjt=0,
                                                                  colorjt_conf=self.data_aug_conf['color_jitter_conf'],)

            mask = self.get_label(mask)
            if True:
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

        return (torch.stack(dep_image), torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans), torch.stack(seg_mask))

    def get_vectors(self, rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']  # 此场景日志信息，采集地名字
        # ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])  # 车本身在世界坐标系位姿
        sd_rec = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])  # 车本身在世界坐标系位姿
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        lidar2ego[:3, 3] = cs_record['translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
        ego2global[:3, 3] = ego_pose['translation']

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        vectors = self.vector_map.gen_vectorized_samples(location, lidar2global_translation,
                                                         lidar2global_rotation)  # lidar2global_translation, lidar2global_rotation



        return vectors

    def get_semantic_map(self, rec):
        vectors= self.get_vectors(rec)
        semantic_mask, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, 3,
                                                                      self.thickness, self.angle_class)


        semantic_masks = semantic_mask != 0
        semantic_iou = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])

        num_cls = semantic_masks.shape[0]
        indices = torch.arange(1, num_cls + 1).reshape(-1, 1, 1)
        semantic_indices = torch.sum(semantic_masks * indices, dim=0)
        # semantic_indices = semantic_indices.unsqueeze(0)
        semantic_indices = torch.flip(semantic_indices,dims=[1])#h w
        semantic_iou = torch.flip(semantic_iou,dims=[2])


        return semantic_indices, semantic_iou

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


        img = np.zeros((self.nx[0], self.nx[1]))
        for name in ['road_segment','lane']:
            for la in lmap[name]:
                pts = np.round((la - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img)


    def get_binimg(self, rec):
        img = np.zeros((self.nx[0], self.nx[1]))

        egopose = self.nusc.get('ego_pose',self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
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

    def get_lidar_data_to_img(self, rec, post_rots, post_trans, cams, downsample, nsweeps):
        points_imgs, _, _ = get_lidar_data_to_img(self.nusc, rec, data_aug_conf=self.data_aug_conf,
                                                  grid_conf=self.grid_conf, post_rots=post_rots, post_trans=post_trans,
                                                  cams=cams, downsample=downsample, nsweeps=nsweeps, min_distance=2.2, )
        return torch.Tensor(points_imgs)
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        depimg,imgs, rots, trans, intrins, post_rots, post_trans,seg_mask = self.get_image_data(rec, cams)
        if self.use_lidar:
            lidar_data= self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)
        else:
            lidar_data = depimg

        index,iou = self.get_semantic_map(rec)
        mask = self.get_binimg(rec)
        road_mask = self.get_static(rec)
        semantica_mask = torch.stack([road_mask,mask])
        binimg = dict()
        binimg['iou'] = iou
        binimg['index'] = index
        binimg['car'] = mask
        binimg['seg_mask'] = semantica_mask
        return imgs, rots, trans, intrins, post_rots, post_trans,lidar_data,binimg,seg_mask

class FlipMappingNuscData(MappingNuscData):
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

            mask_o = np.array(mask_o) + 1
            mask_o = Image.fromarray(mask_o.astype(np.uint8))
            # augmentation (resize, crop, horizontal flip, rotate)
            if self.aug_mode == 'hard':
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
                resize, resize_dims, colorjt,_ = self.sample_augmentation_simple()

                img, post_rot2, post_tran2 = img_transform_weak(img_o, resize, resize_dims, colorjt=colorjt,colorjt_conf=self.data_aug_conf['color_jitter_conf'],)
                mask_o = np.array(mask_o) + 1
                mask_o = Image.fromarray(mask_o.astype(np.uint8))
                mask,_,_ = img_transform_weak(mask_o, resize, resize_dims, colorjt=0,colorjt_conf=self.data_aug_conf['color_jitter_conf'],)
                mask = self.get_label(mask)
                seg_mask.append(mask)
                img_flip, post_rot2_flip, post_tran2_flip = img_transform_simple(img_o, resize, resize_dims)
                mask_o, _, _ = img_transform_simple(mask_o, resize, resize_dims)
                mask_o = self.get_label(mask_o)
                seg_mask_flip.append(mask_o)
                flip=0
                post_rot3, post_tran3 =post_rot2, post_tran2
                img_3 = img
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
                #torch.stack(imgs_3), torch.stack(post_trans_3), torch.stack(post_rots_3),torch.stack(seg_mask_3),
                )

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        depimg, imgs, rots, trans, intrins, post_rots, post_trans, imgs_flip, post_trans_flip, post_rots_flip, seg_mask, seg_mask_o, = self.get_image_data(rec, cams)
        if self.use_lidar:
            lidar = self.get_lidar_data_to_img(rec, post_rots=post_rots, post_trans=post_trans,
                                                    cams=cams, downsample=self.downsample, nsweeps=self.nsweeps)
        else:
            lidar = depimg

        index,iou = self.get_semantic_map(rec)
        mask = self.get_binimg(rec)
        road_mask = self.get_static(rec)
        semantica_mask = torch.stack([road_mask, mask])
        binimg = dict()
        binimg['iou'] = iou
        binimg['index'] = index
        binimg['car'] = mask
        binimg['seg_mask'] = semantica_mask
        return imgs, rots, trans, intrins, post_rots, post_trans,lidar,binimg,imgs_flip,post_rots_flip,post_trans_flip,seg_mask,seg_mask_o
def worker_rnd_init(x):
    np.random.seed(13 + x)

def compile_data_mapping(version, dataroot, data_aug_conf, grid_conf,nsweeps,  domain_gap,source,target,  bsz,nworkers,flip=False):
    nusc = NuScenes(version=version,
                    dataroot=dataroot,
                    verbose=False)

    straindata = MappingNuscData(nusc, is_train=True, data_aug_conf=data_aug_conf,
                        grid_conf=grid_conf, nsweeps=nsweeps,
                        domain_gap=domain_gap, domain=source, domain_type='strain')
    ttraindata =FlipMappingNuscData(nusc, is_train=True, data_aug_conf=data_aug_conf,
                                 grid_conf=grid_conf, nsweeps=nsweeps,
                                 domain_gap=domain_gap, domain=target, domain_type='ttrain')
    tvaldata = MappingNuscData(nusc, is_train=False, data_aug_conf=data_aug_conf,
                      grid_conf=grid_conf, nsweeps=nsweeps,
                      domain_gap=domain_gap, domain=target, domain_type='tval')

    strainloader = torch.utils.data.DataLoader(straindata, batch_size=bsz,
                                               shuffle=True,
                                               num_workers=nworkers,
                                               drop_last=True,
                                               worker_init_fn=worker_rnd_init
                                               )
    ttrainloader = torch.utils.data.DataLoader(ttraindata, batch_size=bsz,
                                               shuffle=True,
                                               num_workers=nworkers,
                                               drop_last=True,
                                               worker_init_fn=worker_rnd_init
                                               )
    tvalloader = torch.utils.data.DataLoader(tvaldata, batch_size=bsz,
                                             shuffle=False,
                                             num_workers=nworkers)
    return strainloader,ttrainloader,tvalloader


def compile_data_mapping_source(version, dataroot, data_aug_conf, grid_conf,nsweeps,  domain_gap,source,target,  bsz,nworkers,flip=False):
    nusc = NuScenes(version=version,
                    dataroot=dataroot,
                    verbose=False)

    straindata = MappingNuscData(nusc, is_train=True, data_aug_conf=data_aug_conf,
                        grid_conf=grid_conf, nsweeps=nsweeps,
                        domain_gap=domain_gap, domain=source, domain_type='strain')

    tvaldata = MappingNuscData(nusc, is_train=False, data_aug_conf=data_aug_conf,
                      grid_conf=grid_conf, nsweeps=nsweeps,
                      domain_gap=domain_gap, domain=target, domain_type='tval')

    strainloader = torch.utils.data.DataLoader(straindata, batch_size=bsz,
                                               shuffle=True,
                                               num_workers=nworkers,
                                               drop_last=True,
                                               worker_init_fn=worker_rnd_init
                                               )
    tvalloader = torch.utils.data.DataLoader(tvaldata, batch_size=bsz,
                                             shuffle=False,
                                             num_workers=nworkers)
    return strainloader,tvalloader

def compile_data_mapping_val(version, dataroot, data_aug_conf, grid_conf,nsweeps,  domain_gap,source,target,  bsz,nworkers,flip=False):
    nusc = NuScenes(version=version,
                    dataroot=dataroot,
                    verbose=False)


    tvaldata = MappingNuscData(nusc, is_train=False, data_aug_conf=data_aug_conf,
                      grid_conf=grid_conf, nsweeps=nsweeps,
                      domain_gap=domain_gap, domain=target, domain_type='tval')


    tvalloader = torch.utils.data.DataLoader(tvaldata, batch_size=bsz,
                                             shuffle=False,
                                             num_workers=nworkers)
    return tvalloader