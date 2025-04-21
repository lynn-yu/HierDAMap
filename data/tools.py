"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from functools import reduce

import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
from PIL import Image, ImageFilter
import cv2
from .hsda import *
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
PV_Color={
    0:(110,110,110),
    1: (166, 206, 227),
    2: (251, 154, 153),
    3:(51, 160, 44),
    4:(227, 26, 28),
    5:(255, 127, 0),
    6:(106, 61, 154),
}
def project_to_image(points: np.ndarray, view: np.ndarray, normalize: bool, keep_depth: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        if keep_depth:
            points = np.concatenate((points[0:2, :] / points[2:3, :], points[2:3, :]), axis=0)
        else:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def get_point_img_single(points_lidar,post_trans,post_rots,depth_bins,downsample,points_imgs,count,fH,fW):
    # Step 5: project to augmented image plane
    points_lidar = np.dot(post_rots.numpy()[count], points_lidar)
    points_lidar = points_lidar + post_trans.numpy()[count].reshape(3, 1)

    # Change to the points_imgs: [N_cam, D, H//downsample, W//downsample] representation
    mask = np.ones(points_lidar.shape[1], dtype=bool)
    mask = np.logical_and(mask, points_lidar[2, :] > (depth_bins[0] - depth_bins[2] / 2.0))
    mask = np.logical_and(mask, points_lidar[2, :] < (depth_bins[1] - depth_bins[2] / 2.0))
    mask = np.logical_and(mask, points_lidar[0, :] > 0)
    mask = np.logical_and(mask, (points_lidar[0, :] // downsample) < fW)
    mask = np.logical_and(mask, points_lidar[1, :] > 0)
    mask = np.logical_and(mask, (points_lidar[1, :] // downsample) < fH)
    # mask = np.logical_and(mask, mask_ori)
    points_lidar = points_lidar[:, mask]

    # fill in the volume
    # points_imgs[count, np.rint(points_lidar[2, :]).astype(int)-np.rint(depth_bins[0]).astype(int),
    #             points_lidar[1, :].astype(int) // downsample, points_lidar[0, :].astype(int) // downsample] += 1
    points_imgs[count, np.rint((points_lidar[2, :] - depth_bins[0]) / depth_bins[2]).astype(int),
    points_lidar[1, :].astype(int) // downsample, points_lidar[0, :].astype(int) // downsample] += 1
    return points_imgs

def get_lidar_data_to_img_group(nusc, sample_rec, data_aug_conf, grid_conf, cams, downsample, nsweeps,
                          min_distance,post_rots_o, post_trans_o,post_rots_i, post_trans_i, ):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((4, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.

        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))

        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)
        new_points = current_pc



        points = np.concatenate((points, new_points.points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    # points [4, N]

    ogfH, ogfW = data_aug_conf['final_dim']
    fH, fW = ogfH // downsample, ogfW // downsample
    depth_bins = np.array(grid_conf['dbound'])
    D = np.rint((depth_bins[1] - depth_bins[0]) / depth_bins[2]).astype(int)  # [4, 5, ..., 44]

    # print((cams, D, fH, fW))
    # Get camera poses and timestamp, project point clouds into camera/image planes
    points_imgs_o = np.zeros((len(cams), D, fH, fW))
    points_imgs_i = np.zeros((len(cams), D, fH, fW))
    points_imgs_s = np.zeros((len(cams), D, fH, fW))
    for count, cam in enumerate(cams):
        cam_rec = nusc.get('sample_data', sample_rec['data'][cam])

        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        cam_calib_rec = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
        cam_pose_rec = nusc.get('ego_pose', cam_rec['ego_pose_token'])

        # Step 1: From ego car to global
        car_to_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                         inverse=False)

        # Step 2: From global to ego
        global_to_cam_ego = transform_matrix(cam_pose_rec['translation'],
                                             Quaternion(cam_pose_rec['rotation']), inverse=True)

        # Step 3: From ego to camera
        cam_ego_to_cam = transform_matrix(cam_calib_rec['translation'],
                                          Quaternion(cam_calib_rec['rotation']), inverse=True)

        # Fuse three transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [cam_ego_to_cam, global_to_cam_ego, car_to_global])
        # points_lidar.transform(trans_matrix)
        points_lidar = trans_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]

        # Step 4: project to image plane
        points_lidar = project_to_image(points_lidar, np.array(cam_calib_rec['camera_intrinsic']),
                                        normalize=True, keep_depth=True)
        #print(count)
        points_imgs_o = get_point_img_single(points_lidar,post_trans_o,post_rots_o,depth_bins,downsample,points_imgs_o,count,fH,fW)

        points_imgs_i = get_point_img_single(points_lidar, post_trans_i, post_rots_i, depth_bins, downsample, points_imgs_i, count,fH,fW)

        #points_imgs_s = get_point_img_single(points_lidar, post_trans_s, post_rots_s, depth_bins, downsample,points_imgs_s, count)


    return points_imgs_o,points_imgs_i

def get_lidar_data_to_img(nusc, sample_rec, data_aug_conf, grid_conf, post_rots, post_trans, cams, downsample, nsweeps,
                          min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((4, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.

        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))

        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)
        new_points = current_pc

        # print(new_points.points, 'Here')
        # exit
        # # Add time vector which can be used as a temporal feature.
        # time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        # times = time_lag * np.ones((1, current_pc.nbr_points()))

        # new_points = np.concatenate((current_pc.points, times), 0)

        points = np.concatenate((points, new_points.points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    # points [4, N]

    ogfH, ogfW = data_aug_conf['final_dim']
    fH, fW = ogfH // downsample, ogfW // downsample
    depth_bins = np.array(grid_conf['dbound'])
    D = np.rint((depth_bins[1] - depth_bins[0]) / depth_bins[2]).astype(int)  # [4, 5, ..., 44]

    # print((cams, D, fH, fW))
    # Get camera poses and timestamp, project point clouds into camera/image planes
    points_imgs = np.zeros((len(cams), D, fH, fW))
    ori_points_imgs = np.zeros((len(cams), fH, fW))
    relative_depth = []
    for count, cam in enumerate(cams):

        cam_rec = nusc.get('sample_data', sample_rec['data'][cam])

        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        cam_calib_rec = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
        cam_pose_rec = nusc.get('ego_pose', cam_rec['ego_pose_token'])

        # Step 1: From ego car to global
        car_to_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                         inverse=False)

        # Step 2: From global to ego
        global_to_cam_ego = transform_matrix(cam_pose_rec['translation'],
                                             Quaternion(cam_pose_rec['rotation']), inverse=True)

        # Step 3: From ego to camera
        cam_ego_to_cam = transform_matrix(cam_calib_rec['translation'],
                                          Quaternion(cam_calib_rec['rotation']), inverse=True)

        # Fuse three transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [cam_ego_to_cam, global_to_cam_ego, car_to_global])
        # points_lidar.transform(trans_matrix)
        points_lidar = trans_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]

        # Step 4: project to image plane
        points_lidar = project_to_image(points_lidar, np.array(cam_calib_rec['camera_intrinsic']),
                                        normalize=True, keep_depth=True)



        # Step 5: project to augmented image plane
        points_lidar = np.dot(post_rots.numpy()[count], points_lidar)
        points_lidar = points_lidar + post_trans.numpy()[count].reshape(3, 1)
        ###get original depth
        mask = np.ones(points_lidar.shape[1], dtype=bool)
        mask = np.logical_and(mask, points_lidar[2, :] > (depth_bins[0] - depth_bins[2] / 2.0))
        mask = np.logical_and(mask, points_lidar[2, :] < (depth_bins[1] - depth_bins[2] / 2.0))
        mask = np.logical_and(mask, points_lidar[0, :] > 0)
        mask = np.logical_and(mask, (points_lidar[0, :] // downsample) < fW)
        mask = np.logical_and(mask, points_lidar[1, :] > 0)
        mask = np.logical_and(mask, (points_lidar[1, :] // downsample) < fH)
        points_lidar_ori = points_lidar[:, mask]

        # Change to the points_imgs: [N_cam, D, H//downsample, W//downsample] representation
        mask = np.ones(points_lidar.shape[1], dtype=bool)
        mask = np.logical_and(mask, points_lidar[2, :] > (depth_bins[0] - depth_bins[2] / 2.0))
        mask = np.logical_and(mask, points_lidar[2, :] < (depth_bins[1] - depth_bins[2] / 2.0))
        mask = np.logical_and(mask, points_lidar[0, :] > 0)
        mask = np.logical_and(mask, (points_lidar[0, :] // downsample) < fW)
        mask = np.logical_and(mask, points_lidar[1, :] > 0)
        mask = np.logical_and(mask, (points_lidar[1, :] // downsample) < fH)
        # mask = np.logical_and(mask, mask_ori)
        points_lidar = points_lidar[:, mask]

        # fill in the volume
        # points_imgs[count, np.rint(points_lidar[2, :]).astype(int)-np.rint(depth_bins[0]).astype(int),
        #             points_lidar[1, :].astype(int) // downsample, points_lidar[0, :].astype(int) // downsample] += 1
        points_imgs[count, np.rint((points_lidar[2, :] - depth_bins[0]) / depth_bins[2]).astype(int),
        points_lidar[1, :].astype(int) // downsample, points_lidar[0, :].astype(int) // downsample] += 1
        ori_points_imgs[count, points_lidar_ori[1, :].astype(int) // downsample, points_lidar_ori[0, :].astype(
            int) // downsample] = points_lidar_ori[2, :]
        real_dep = torch.tensor(points_imgs[count])  # d h w
        _, real_dep = torch.max(real_dep, dim=0)  # h w
        real_dep = (real_dep - real_dep.min()) / (
                    real_dep.max() - real_dep.min())  # (1 - (real_dep - real_dep.min()) / (real_dep.max() - real_dep.min()))#
        relative_depth.append(real_dep)

    return points_imgs, torch.tensor(ori_points_imgs), torch.stack(relative_depth)
def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,gauss,
                  resize, resize_dims, crop,
                  flip, rotate, colorjt, colorjt_conf):

    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if gauss:
        img = img.filter(ImageFilter.GaussianBlur(radius = gauss))
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

    img = img.rotate(rotate)
    if colorjt:
        img = torchvision.transforms.ColorJitter(brightness=colorjt_conf[0], contrast=colorjt_conf[1],
                                                 saturation=colorjt_conf[2], hue=colorjt_conf[3])(img)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b
    return img, post_rot, post_tran





def img_transform_weak(img, resize, resize_dims, colorjt, colorjt_conf):
    post_rot2 = torch.eye(2)
    post_tran2 = torch.zeros(2)

    img = img.resize(resize_dims)
    if colorjt:
        img = torchvision.transforms.ColorJitter(brightness=colorjt_conf[0], contrast=colorjt_conf[1],
                                                 saturation=colorjt_conf[2], hue=colorjt_conf[3])(img)
    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2#矩阵乘法
    post_tran2 = rot_resize @ post_tran2


    return img, post_rot2, post_tran2

def img_transform_simple(img, resize, resize_dims):
    post_rot2 = torch.eye(2)
    post_tran2 = torch.zeros(2)

    img = img.resize(resize_dims)

    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2#矩阵乘法
    post_tran2 = rot_resize @ post_tran2


    return img, post_rot2, post_tran2
def img_transform_strong(img, resize, resize_dims, flip=False, colorjt=False,colorjt_conf=None,rotate=0,gauss=False):
    post_rot2 = torch.eye(2)
    post_tran2 = torch.zeros(2)

    dep_img = img.resize((196, 364))
    img = img.resize(resize_dims)

    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)
    if colorjt:
        img = torchvision.transforms.ColorJitter(brightness=colorjt_conf[0], contrast=colorjt_conf[1],
                                                 saturation=colorjt_conf[2], hue=colorjt_conf[3])(img)
    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2#矩阵乘法
    post_tran2 = rot_resize @ post_tran2
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([resize_dims[0], 0])
        post_rot2 = A.matmul(post_rot2)
        post_tran2 = A.matmul(post_tran2) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([resize_dims[0], resize_dims[1]]) / 2
    b = A.matmul(-b) + b
    post_rot2 = A.matmul(post_rot2)
    post_tran2 = A.matmul(post_tran2) + b
    post_tran = torch.zeros(2)
    post_rot = torch.eye(2)
    post_tran[:2] = post_tran2
    post_rot[:2, :2] = post_rot2

    return img, post_rot, post_tran


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


denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))
ori_normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()
    return {
            'loss': total_loss / len(valloader.dataset),
            'iou': total_intersect / total_union,
            }


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys
