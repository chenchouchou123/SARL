import os
import argparse

import pickle
import torch
import smplx
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import json
# import pickle
from .LiDAR_dataset import lidar_Dataset

def camera_to_pixel(X, intrinsics, distortion_coefficients):
    # focal length
    f = intrinsics[:2]
    # center principal point
    c = intrinsics[2:]
    k = np.array([distortion_coefficients[0],
                 distortion_coefficients[1], distortion_coefficients[4]])
    p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
    XX = X[..., :2] / (X[..., 2:])
    # XX = pd.to_numeric(XX, errors='coere')
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

    radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                        axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c

def world_to_pixels(X, extrinsic_matrix, cam):
    B, N, dim = X.shape
    X = np.concatenate((X, np.ones((B, N, 1))), axis=-1).transpose(0, 2, 1)
    X = (extrinsic_matrix @ X).transpose(0, 2, 1)
    X = camera_to_pixel(X[..., :3].reshape(B*N, dim), cam['intrinsics'], [0]*5)
    X = X.reshape(B, N, -1)
    
    def check_pix(p):
        rule1 = p[:, 0] > 0
        rule2 = p[:, 0] < cam['width']
        rule3 = p[:, 1] > 0
        rule4 = p[:, 1] < cam['height']
        rule  = [a and b and c and d for a, b, c, d in zip(rule1, rule2, rule3, rule4)]
        return p[rule] if len(rule) > 50 else []
    
    X = [check_pix(xx) for xx in X]

    return X

def get_bool_from_coordinates(coordinates, shape=(1080, 1920)):
    bool_arr = np.zeros(shape, dtype=bool)
    if len(coordinates) > 0:
        bool_arr[coordinates[:, 0], coordinates[:, 1]] = True

    return bool_arr

def fix_points_num(points: np.array, num_points: int):
    """
    downsamples the points using voxel and uniform downsampling, 
    and either repeats or randomly selects points to reach the desired number.
    
    Args:
      points (np.array): a numpy array containing 3D points.
      num_points (int): the desired number of points 
    
    Returns:
      a numpy array `(num_points, 3)`
    """
    # print(points.shape)
    if len(points) == 0:
        return np.zeros((num_points, 3))
    points = points[~np.isnan(points).any(axis=-1)]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc = pc.voxel_down_sample(voxel_size=0.05)
    ratio = int(len(pc.points) / num_points + 0.05)
    if ratio > 1:
        pc = pc.uniform_down_sample(ratio)

    points = np.asarray(pc.points)
    origin_num_points = points.shape[0]

    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    else:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res

INTRINSICS = [599.628, 599.466, 971.613, 540.258]
DIST       = [0.003, -0.003, -0.001, 0.004, 0.0]
LIDAR2CAM  = [[[-0.0355545576, -0.999323133, -0.0094419378, -0.00330376451], 
              [0.00117895777, 0.00940596282, -0.999955068, -0.0498469479], 
              [0.999367041, -0.0355640917, 0.00084373493, -0.0994979365], 
              [0.0, 0.0, 0.0, 1.0]]]

import os
import pickle
import json
import numpy as np
import torch
import open3d as o3d
from smplx import SMPL  # 引入SMPL模型库
from scipy.spatial import cKDTree

class SLOPER4D_Dataset(lidar_Dataset):
    def __init__(self, root_folder, scene_list=[], dataset_path='./Extra/sloper4d/', is_train=True,
                 device='cpu', return_torch=True, fix_pts_num=False, print_info=True,
                 return_smpl=False, augmentation=False, interval=4, load_v2v=False,
                 complete_missing_parts=False):  # 新增补全选项
        super().__init__(is_train=is_train, return_torch=return_torch, fix_pts_num=fix_pts_num,
                         augmentation=augmentation, load_v2v=load_v2v, interval=interval)
        self.root_folder = root_folder
        self.scene_list = scene_list
        self.device = device
        self.return_torch = return_torch
        self.print_info = print_info
        self.fix_pts_num = fix_pts_num
        self.point_num = 1024
        self.return_smpl = return_smpl
        self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
        self.num_joints = 15
        self.bone_index = [[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [5,7],
                           [6,8], [0,9], [9,10], [9,11], [9,12], [11,13], [12,14], [13,15], [14,16]]
        self.augmentation = augmentation
        self.is_train = is_train
        self.complete_missing_parts = complete_missing_parts  # 是否启用点云补全

        # 加载SMPL模型
        self.smpl_model = SMPL(model_path='smplx_models/smpl/SMPL_NEUTRAL.pkl', gender='NEUTRAL')
        self.smpl_model = self.smpl_model.to(device)

        split = 'training' if is_train else 'validation'
        data_file = 'train.pkl' if is_train else 'test.pkl'
        os.makedirs(dataset_path, exist_ok=True)
        data_pkl_file = os.path.join(dataset_path, data_file)

        if not os.path.exists(data_pkl_file):
            scene_data_list = []
            for scene in scene_list:
                pkl_file = os.path.join(root_folder, scene, scene + '_labels.pkl')
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                length = data['total_frames'] if 'total_frames' in data else len(data['frame_num'])
                world2lidar, _ = self.get_lidar_data(data, length)
                self.smpl_pose, self.global_trans, self.betas, self.smpl_gender, self.human_points = self.load_3d_data(data, length)
                fb, lt, bbox, s2d, cp, cam, sv, sj, sm = self.load_rgb_data(data)
                masks = self.load_mask(pkl_file, length)
                smpl_joints = sj[:, self.JOINTS_IDX]
                pc_dist = []
                for ind, sj_ in enumerate(smpl_joints):
                    human_points_this = self.human_points[ind]
                    if len(human_points_this) >= 10:
                        dist1, idx1, dist2, idx2, pc_d = self.nn_distance(human_points_this, sj_, is_cuda=True)
                        pc_d = pc_d.squeeze().detach().cpu().numpy()
                        pc_dist.append(pc_d)
                    else:
                        pc_dist.append([])
                scene_data_list.append({
                    'scene_name': scene,
                    'smpl_pose': self.smpl_pose,
                    'global_trans': self.global_trans,
                    'betas': self.betas,
                    'smpl_gender': self.smpl_gender,
                    'human_points': self.human_points,
                    'file_basename': fb,
                    'lidar_tstamps': lt,
                    'pc_dist': pc_dist,
                    'bbox': bbox,
                    'skel_2d': s2d,
                    'cam_pose': cp,
                    'cam': cam,
                    'smpl_verts': sv,
                    'smpl_joints': smpl_joints,
                    'smpl_mask': sm,
                    'length': length,
                    'world2lidar': world2lidar,
                    'masks': masks
                })
            with open(data_pkl_file, 'wb') as f:
                pickle.dump(scene_data_list, f)
            self.scene_data_list = scene_data_list
        else:
            with open(data_pkl_file, 'rb') as f:
                self.scene_data_list = pickle.load(f)

        self.valid_hkps = []
        self.valid_joints_def = {
            'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'left_knee': 3, 'right_knee': 4,
            'left_ankle': 5, 'right_ankle': 6, 'neck': 7, 'head': 8, 'left_shoulder': 9,
            'right_shoulder': 10, 'left_elbow': 11, 'right_elbow': 12, 'left_wrist': 13, 'right_wrist': 14
        }
        for inds, scene in enumerate(self.scene_data_list):
            for ind, hps in enumerate(scene['human_points']):
                if isinstance(hps, list) or hps.shape[0] < 10:
                    continue
                scene_this = self.scene_data_list[inds]
                human_points = scene_this['human_points'][ind]
                smpl_verts = scene_this['smpl_verts'][ind]
                smpl_joints = scene_this['smpl_joints'][ind]
                mesh_dict = {
                    'body_pose': scene_this['smpl_pose'][ind][3:],
                    'transl': scene_this['global_trans'][ind],
                    'betas': scene_this['betas'],
                    'global_orient': scene_this['smpl_pose'][ind][:3]
                }

                # 点云补全逻辑
                if self.complete_missing_parts:
                    human_points = self.complete_point_cloud(human_points, mesh_dict, smpl_joints)
                    # 更新scene_data_list中的human_points
                    self.scene_data_list[inds]['human_points'][ind] = human_points

                self.valid_hkps.append({
                    'mesh_dict': mesh_dict,
                    'smpl_joints': smpl_joints,
                    'smpl_verts': smpl_verts,
                    'human_points': human_points,
                    'pc_dist': scene_this['pc_dist'][ind],
                    'location_dict': {'scene': self.scene_list[inds], 'time': str(ind), 'id': 0}
                })

        self.load_v2v = load_v2v
        if load_v2v:
            self.v2v_pred = {}
            for scene in self.scene_list:
                with open(os.path.join('pose_results', 'v2v', 'sloper4d', scene + '.json'), 'r') as f:
                    data_ = json.load(f)
                    for key in data_.keys():
                        data_[key] = np.array(data_[key])
                    self.v2v_pred[scene] = data_


        self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
        with open('smplx_models/smpl/SMPL_NEUTRAL.pkl', 'rb') as smpl_file:
            self.joint_24_regressor = torch.tensor(pickle.load(smpl_file, encoding='latin1')['J_regressor'].todense()).float()
        self.interval = interval

    def complete_point_cloud(self, human_points, mesh_dict, smpl_joints):
        """
        使用SMPL模型补全缺失的点云（手部和脚部）。
        
        Args:
            human_points: numpy array, 原始点云 (N, 3)
            mesh_dict: dict, 包含SMPL参数（body_pose, transl, betas, global_orient）
            smpl_joints: numpy array, SMPL关节位置 (num_joints, 3)
        
        Returns:
            completed_points: numpy array, 补全后的点云
        """
        # 将点云和关节转换为torch tensor
        human_points_torch = torch.from_numpy(human_points).float().to(self.device)
        smpl_joints_torch = torch.from_numpy(smpl_joints).float().to(self.device)

        # 检查缺失区域（以手部和脚部为例）
        kdtree = cKDTree(human_points)
        hand_joints = [self.valid_joints_def['left_wrist'], self.valid_joints_def['right_wrist']]  # 手部关节
        foot_joints = [self.valid_joints_def['left_ankle'], self.valid_joints_def['right_ankle']]  # 脚部关节
        missing_regions = []

        for joint_idx in hand_joints + foot_joints:
            joint_pos = smpl_joints[joint_idx]
            dists, _ = kdtree.query(joint_pos, k=1)
            if dists > 0.1:  # 假设距离大于0.1米表示缺失
                missing_regions.append(joint_idx)

        if not missing_regions:
            return human_points  # 无缺失，直接返回

        # 使用SMPL生成完整网格
        body_pose = torch.from_numpy(mesh_dict['body_pose']).float().to(self.device).unsqueeze(0)
        global_orient = torch.from_numpy(mesh_dict['global_orient']).float().to(self.device).unsqueeze(0)
        betas = torch.from_numpy(np.array(mesh_dict['betas'])).float().to(self.device).unsqueeze(0)
        
        transl = torch.from_numpy(np.array(mesh_dict['transl'])).float().to(self.device).unsqueeze(0)

        smpl_output = self.smpl_model(
            body_pose=body_pose,
            global_orient=global_orient,
            betas=betas,
            transl=transl,
            return_verts=True
        )
        smpl_vertices = smpl_output.vertices[0].detach().cpu().numpy()  # (6890, 3)

        # 定义手部和脚部的顶点索引（需要根据SMPL模型的顶点拓扑定义）
        # 这里仅为示例，实际需要根据SMPL模型的顶点索引表确定
        hand_vertex_indices = np.arange(6000, 6500)  # 假设手部顶点索引
        foot_vertex_indices = np.arange(6500, 6890)  # 假设脚部顶点索引
        region_indices = {
            self.valid_joints_def['left_wrist']: hand_vertex_indices,
            self.valid_joints_def['right_wrist']: hand_vertex_indices,
            self.valid_joints_def['left_ankle']: foot_vertex_indices,
            self.valid_joints_def['right_ankle']: foot_vertex_indices
        }

        # 提取缺失区域的点云
        completed_points = human_points.copy()
        for joint_idx in missing_regions:
            region_verts = smpl_vertices[region_indices[joint_idx]]
            # 随机采样与原始点云密度相当的点
            sample_size = min(len(region_verts), 100)  # 假设每个区域采样100个点
            sampled_indices = np.random.choice(len(region_verts), sample_size, replace=False)
            sampled_points = region_verts[sampled_indices]
            completed_points = np.vstack([completed_points, sampled_points])

        return completed_points

# class SLOPER4D_Dataset(lidar_Dataset):
#     def __init__(self, root_folder, scene_list = [], dataset_path = './Extra/sloper4d/', is_train = True,
#                  device='cpu', 
#                  return_torch:bool=True, 
#                  fix_pts_num:bool=False,
#                  print_info:bool=True,
#                  return_smpl:bool=False,
#                  augmentation:bool=False, interval = 4, load_v2v = False):
#         super().__init__(is_train = is_train,
#                  return_torch=return_torch, 
#                  fix_pts_num=fix_pts_num,
#                  augmentation=augmentation,
#                  load_v2v = load_v2v, 
#                  interval = interval)
#         self.root_folder = root_folder
#         self.scene_list = scene_list
#         self.device       = device
#         self.return_torch = return_torch
#         self.print_info   = print_info
#         self.fix_pts_num  = fix_pts_num
#         self.point_num = 1024
#         self.return_smpl  = return_smpl
#         # self.joint_index = np.array([0,1,2,4,5,7,8,10,11,12,15,16,17,18,19,20,21])
#         self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
#         self.num_joints = 15
#         self.bone_index = [[0,1], [0,2], [1,3], [2,4], [3,5], [4,6], [5,7], \
#             [6,8], [0,9], [9,10], [9,11], [9,12], [11,13], [12,14], [13,15], [14,16]]
#         self.augmentation = augmentation
#         self.is_train = is_train
#         split = 'training' if is_train else 'validation'
#         data_file = 'train.pkl' if is_train else 'test.pkl'
#         os.makedirs(dataset_path, exist_ok=True)
#         data_pkl_file = os.path.join(dataset_path, data_file)

#         if not os.path.exists(data_pkl_file):
#             scene_data_list = []
#             for scene in scene_list:
#                 pkl_file = os.path.join(root_folder, scene, scene + '_labels.pkl')
#                 with open(pkl_file, 'rb') as f:
#                     data = pickle.load(f)
#                 length = data['total_frames'] if 'total_frames' in data else len(data['frame_num'])
#                 world2lidar, _ = self.get_lidar_data(data, length)
#                 self.smpl_pose, self.global_trans, self.betas, self.smpl_gender, self.human_points = self.load_3d_data(data, length)   
#                 fb, lt, bbox, s2d, cp, cam, sv, sj, sm = self.load_rgb_data(data) 
#                 masks = self.load_mask(pkl_file, length)
#                 smpl_joints = sj[:,self.JOINTS_IDX]
#                 pc_dist = []
#                 for ind, sj_ in enumerate(smpl_joints):
#                     human_points_this = self.human_points[ind]
#                     if len(human_points_this) >= 10:
#                         dist1, idx1, dist2, idx2, pc_d = self.nn_distance(human_points_this, sj_, is_cuda=True)
#                         pc_d = pc_d.squeeze().detach().cpu().numpy()
#                         pc_dist.append(pc_d)
#                     else:
#                         pc_dist.append([])
#                 scene_data_list.append({
#                     'scene_name':scene,
#                     'smpl_pose': self.smpl_pose,
#                     'global_trans': self.global_trans,
#                     'betas': self.betas,
#                     'smpl_gender': self.smpl_gender,
#                     'human_points': self.human_points,
#                     'file_basename': fb,
#                     'lidar_tstamps': lt,
#                     'pc_dist': pc_dist,
#                     'bbox': bbox,
#                     'skel_2d': s2d,
#                     'cam_pose': cp,
#                     'cam': cam,
#                     'smpl_verts': sv,
#                     'smpl_joints': smpl_joints,
#                     'smpl_mask': sm,
#                     'length': length,
#                     'world2lidar': world2lidar,
#                     'masks' : masks
#                 })
#             with open(data_pkl_file, 'wb') as f:
#                 pickle.dump(scene_data_list, f)
#             self.scene_data_list = scene_data_list
#         else:
#             with open(data_pkl_file, 'rb') as f:
#                 self.scene_data_list = pickle.load(f)
#             #print(self.scene_data_list)

#         self.valid_hkps = []
#         for inds, scene in enumerate(self.scene_data_list):
#             for ind, hps in enumerate(scene['human_points']):
#                 if type(hps)!=list and hps.shape[0] >= 10:
#                     scene_this = self.scene_data_list[inds]
#                     human_points = scene_this['human_points'][ind]
#                     smpl_verts = scene_this['smpl_verts'][ind]
#                     smpl_joints = scene_this['smpl_joints'][ind]
#                     mesh_dict = {}
#                     mesh_dict['body_pose'] = scene_this['smpl_pose'][ind][3:]
#                     mesh_dict['transl'] = scene_this['global_trans'][ind]
#                     mesh_dict['betas'] = scene_this['betas']
#                     mesh_dict['global_orient'] = scene_this['smpl_pose'][ind][:3]
#                     self.valid_hkps.append({'mesh_dict':mesh_dict, 'smpl_joints':smpl_joints, \
#                                                 'smpl_verts':smpl_verts,'human_points':human_points,\
#                                                       'pc_dist':scene_this['pc_dist'][ind], 'location_dict': {'scene':self.scene_list[inds], 'time':str(ind), 'id': 0}})
#         self.load_v2v = load_v2v
#         if load_v2v:
#             self.v2v_pred = {}
#             for scene in self.scene_list:
#                 with open(os.path.join('pose_results', 'v2v', 'sloper4d', scene + '.json'), 'r') as f:
#                     data_ = json.load(f)
#                     # import pdb; pdb.set_trace()
#                     # self.v2v_pred[scene] = json.load(f)
#                     for key in data_.keys():
#                         data_[key] = np.array(data_[key])
#                     self.v2v_pred[scene] = data_

#         self.valid_joints_def = {
#             'pelvis':0,
#             'left_hip':1,
#             'right_hip':2,
#             'left_knee':3,
#             'right_knee':4,
#             'left_ankle':5,
#             'right_ankle':6,
#             'neck':7,
#             'head':8,
#             'left_shoulder':9,
#             'right_shoulder':10,
#             'left_elbow':11,
#             'right_elbow':12,
#             'left_wrist':13,
#             'right_wrist':14
#             }
#         self.JOINTS_IDX = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
#         with open('smplx_models/smpl/SMPL_NEUTRAL.pkl', 'rb') as smpl_file:
#             self.joint_24_regressor = torch.tensor(pickle.load(smpl_file, encoding='latin1')['J_regressor'].todense()).float()
#         # self.length = sum([scene['length'] for scene in self.scene_data_list])
#         self.interval = interval

    def get_lidar_data(self, data, length, is_inv=True):
        lidar_traj    = data['first_person']['lidar_traj'].copy()
        lidar_tstamps = lidar_traj[:length, -1]
        world2lidar   = np.array([np.eye(4)] * length)
        world2lidar[:, :3, :3] = R.from_quat(lidar_traj[:length, 4: 8]).inv().as_matrix()
        world2lidar[:, :3, 3:] = -world2lidar[:, :3, :3] @ lidar_traj[:length, 1:4].reshape(-1, 3, 1)

        return world2lidar, lidar_tstamps
    
    def load_rgb_data(self, data):
        try:
            cam = data['RGB_info']     
        except:
            print('=====> Load default camera parameters.')
            cam = {'fps':20, 'width': 1920, 'height':1080, 
                        'intrinsics':INTRINSICS, 'lidar2cam':LIDAR2CAM, 'dist':DIST}
            
        file_basename = data['RGB_frames']['file_basename'] # synchronized img file names
        lidar_tstamps = data['RGB_frames']['lidar_tstamps'] # synchronized ldiar timestamps
        bbox          = data['RGB_frames']['bbox']          # 2D bbox of the tracked human (N, [x1, y1, x2, y2])
        skel_2d       = data['RGB_frames']['skel_2d']       # 2D keypoints (N, [17, 3]), every joint is (x, y, probability)
        cam_pose      = data['RGB_frames']['cam_pose']      # extrinsic, world to camera (N, [4, 4])

        if self.return_smpl:
            smpl_verts, smpl_joints = self.return_smpl_verts()
            smpl_mask = world_to_pixels(smpl_verts, cam_pose, cam)
            return file_basename, lidar_tstamps, bbox, skel_2d, cam_pose, cam, smpl_verts, smpl_joints, smpl_mask
        else:
            return file_basename, lidar_tstamps, bbox, skel_2d, cam_pose, cam, None, None, None

    def load_mask(self, pkl_file, length):
        mask_pkl = pkl_file[:-4] + "_mask.pkl"
        if os.path.exists(mask_pkl):
            with open(mask_pkl, 'rb') as f:
                print(f'Loading: {mask_pkl}')
                masks = pickle.load(f)['masks']
        else:
            masks = [[]]* length
        return masks

    def load_3d_data(self, data, length, person='second_person', points_num = 1024):
        assert length <= len(data['frame_num']), f"RGB length must be less than point cloud length"
        point_clouds = [[]] * length
        if 'point_clouds' in data[person]:
            for i, pf in enumerate(data[person]['point_frame']):
                index = data['frame_num'].index(pf)
                if index < length:
                    point_clouds[index] = data[person]['point_clouds'][i]
        if False:
            point_clouds = np.array([fix_points_num(pts, points_num) for pts in point_clouds])

        sp = data['second_person']
        smpl_pose    = sp['opt_pose'][:length].astype(np.float32)  # n x 72 array of scalars
        global_trans = sp['opt_trans'][:length].astype(np.float32) # n x 3 array of scalars
        betas        = sp['beta']                                       # n x 10 array of scalars
        smpl_gender  = sp['gender']                                     # male/female/neutral
        human_points = point_clouds                                     # list of n arrays, each of shape (x_i, 3)
        return smpl_pose, global_trans, betas, smpl_gender, human_points

    def updata_pkl(self, img_name, 
                   bbox=None, 
                   cam_pose=None, 
                   keypoints=None):
        if img_name in self.file_basename:
            index = self.file_basename.index(img_name)
            if bbox is not None:
                self.data['RGB_frames']['bbox'][index] = bbox
            if keypoints is not None:
                self.data['RGB_frames']['skel_2d'][index] = keypoints
            if cam_pose is not None:
                self.data['RGB_frames']['cam_pose'][index] = cam_pose
        else:
            print(f"{img_name} is not in the synchronized labels file")
    
    def get_rgb_frames(self, ):
        return self.data['RGB_frames']

    def save_pkl(self, overwrite=False):
        
        save_path = self.pkl_file if overwrite else self.pkl_file[:-4] + '_updated.pkl' 
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"{save_path} saved")

    def check_length(self):
        # Check if all the lists inside rgb_frames have the same length
        assert all(len(lst) == self.length for lst in [self.bbox, self.skel_2d,  
                                                       self.lidar_tstamps, self.masks, 
                                                       self.smpl_pose, self.global_trans, 
                                                       self.human_points])

        print(f'Data length: {self.length}')
        
    def get_cam_params(self): 
        return torch.from_numpy(np.array(self.cam['lidar2cam']).astype(np.float32)).to(self.device), \
               torch.from_numpy(np.array(self.cam['intrinsics']).astype(np.float32)).to(self.device), \
               torch.from_numpy(np.array(self.cam['dist']).astype(np.float32)).to(self.device)
            
    def get_img_shape(self):
        return self.cam['width'], self.cam['height']

    def return_smpl_verts(self, ):
        file_path = self.root_folder
        with torch.no_grad():
            human_model = smplx.create('./smplx_models/', model_type = 'smpl',
                                    gender=self.smpl_gender, 
                                    use_face_contour=False,
                                    ext="npz")
            orient = torch.tensor(self.smpl_pose).float()[:, :3]
            bpose  = torch.tensor(self.smpl_pose).float()[:, 3:]
            transl = torch.tensor(self.global_trans).float()
            smpl_md = human_model(betas=torch.tensor(self.betas).reshape(-1, 10).float(), 
                                    return_verts=True, 
                                    body_pose=bpose,
                                    global_orient=orient,
                                    transl=transl)
            
        return smpl_md.vertices.numpy(), smpl_md.joints.numpy()#[:,self.joint_index,:]
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D dataset')
    parser.add_argument('--dataset_root', type=str, 
                        default='/Extra/fanbohao/posedataset/PointC/sloper4d/', 
                        help='Path to data file')
    parser.add_argument('--scene_name', type=str, 
                        default='seq003_street_002', 
                        help='Scene name')
    # parser.add_argument('--pkl_file', type=str, 
    #                     default='/disk1/fanbohao/fbh_data/sloper4d/seq003_street_002/seq003_street_002_labels.pkl', 
    #                     help='Path to the pkl file')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='The batch size of the data loader')
    parser.add_argument('--index', type=int, default=-1,
                        help='the index frame to be saved to a image')
    args = parser.parse_args()
    scene_train = [
        'seq002_football_001',
        'seq003_street_002',
        'seq005_library_002',
        'seq007_garden_001',
        'seq008_running_001'
    ]
    scene_test = ['seq009_running_002']
    train_dataset = SLOPER4D_Dataset(args.dataset_root, scene_train, is_train = True, dataset_path = './save_data/sloper4d',
                               return_torch=False, 
                               fix_pts_num=True, return_smpl = True)
    test_dataset = SLOPER4D_Dataset(args.dataset_root, scene_test, is_train = False, dataset_path = './save_data/sloper4d',
                               return_torch=False, 
                               fix_pts_num=True, return_smpl = True)
    # import pdb; pdb.set_trace()
    #
    # =====> attention 
    # Batch_size > 1 is not supported yet
    # because bbox and 2d keypoints missing in some frames
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    for index, sample in enumerate(dataloader):
        # import pdb; pdb.set_trace()
        human_points_local = sample['human_points_local'][0].cpu()
        smpl_joints_local = sample['smpl_joints_local'][0].cpu()
        # human_points_local = sample['human_points'][0].cpu()
        # smpl_joints_local = sample['smpl_joints'][0].cpu()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(human_points_local[:,0], human_points_local[:,1], human_points_local[:,2], c = 'gray', s = 1)
        ax.scatter(smpl_joints_local[:,0], smpl_joints_local[:,1], smpl_joints_local[:,2], c = 'r', s = 5)
        for i in range(smpl_joints_local.shape[0]):
            ax.text(smpl_joints_local[i,0], smpl_joints_local[i,1], smpl_joints_local[i,2], s = str(i))
        x,y,z = smpl_joints_local[0]
        ax.set_xlim(x-1.0, x+1.0)
        ax.set_ylim(y-1.0, y+1.0)
        ax.set_zlim(z-1.0, z+1.0)
        # print(smpl_joints_local[[29,31],:].mean(dim = 0), smpl_joints_local[[32,34],:].mean(dim = 0),)
        plt.show()
