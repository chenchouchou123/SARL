import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datasets.sloper4d import SLOPER4D_Dataset
from datasets.humanm3 import HumanM3_Dataset

from models.framework.SarL import SarL
# from models.v2v_posenet import V2VPoseNet
from tqdm import tqdm
import torch.optim as optim
import logging
import torch.nn.functional as F
import argparse
# import torch.distributed as dist
from scripts.eval_utils import mean_per_vertex_error, mean_per_joint_position_error, reconstruction_error, all_gather, setup_seed, joint_accuracy

from fvcore.nn import FlopCountAnalysis, parameter_count_table

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--model', default='', required=False, type=str)
    parser.add_argument(
        '--dataset', default='sloper4d', required=False, type=str)
    parser.add_argument(
        '--state_dict', default='', required=False, type=str)
    parser.add_argument(
        '--cfg', default='', required=False, type=str)
    args, rest = parser.parse_known_args()
    return args

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, dataloader, optimizer, epoch):
    model.train()
    loss_all = AverageMeter()
    loss_seg = AverageMeter()
    loss_joint = AverageMeter()
    loss_prior = AverageMeter()
    PRINT_FREQ = 50
    for i, sample in enumerate(dataloader):
        pcd = sample['human_points_local']
        # center = sample['global_trans']
        # optimizer.zero_grad()
        # flops = FlopCountAnalysis(model, pcd)
        # print("FLOPs: ", flops.total())
        
        ret_dict = model(pcd)
        loss_dict = model.all_loss(ret_dict, sample)
        all_loss = loss_dict['loss']#loss_dict['loss']
        all_loss.backward()
        optimizer.step()
        loss_all.update(loss_dict['loss'].item())
        loss_joint.update(loss_dict['loss_joint'].item())
        loss_seg.update(loss_dict['loss_seg'].item())
        if 'loss_prior' in loss_dict.keys():
            loss_prior.update(loss_dict['loss_prior'].item())
        if i % PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss_all: {loss_all.val:.3f} ({loss_all.avg:.3f})\t' \
                  'loss_seg: {loss_seg.val:.3f} ({loss_seg.avg:.3f})\t' \
                    'loss_joint: {loss_joint.val:.3f} ({loss_joint.avg:.3f})\t' \
                    'loss_prior: {loss_prior.val:.3f} ({loss_prior.avg:.3f})'.format(
                    epoch, i, len(dataloader), loss_all=loss_all, loss_seg=loss_seg, \
                        loss_joint = loss_joint, loss_prior = loss_prior)
            print(msg)
            # logger.info(msg)





import torch
import time
from tqdm import tqdm
from thop import profile, clever_format
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def test(model, dataloader):
    model.eval()
    mpjpe = 0
    number = 0
    pa_mpjpe = 0
    total_inference_time = 0
    joint_mAP = torch.zeros(15).cuda()
    joint_mpjpe = torch.zeros(15).cuda()
    total_mAP = 0
    total_mpjpe = 0


    dummy_input = torch.randn(1, dataloader.dataset[0]['human_points_local'].shape[0],
                              dataloader.dataset[0]['human_points_local'].shape[1]).cuda()

    macs, params = profile(model, inputs=(dummy_input,), verbose=False, custom_ops={torch.nn.ReLU: None})
    macs, params = clever_format([macs, params], "%.2f")

    if 'M' in params:
        params = float(params.replace('M', ''))
    elif 'K' in params:
        params = float(params.replace('K', '')) / 1024


    print('============Testing============')
    show = True

    for sample in tqdm(dataloader):
        n=0
        for key in sample:
            if type(sample[key]) is not dict and type(sample[key]) is not list:
                sample[key] = sample[key].cuda()
            n+=1
        pcd = sample['human_points_local']
        if model.pose_dim == 24:
            sample['smpl_joints_local'] = sample['smpl_joints24_local']
        gt_pose = sample['smpl_joints_local']

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            ret_dict = model(pcd)

        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        pose = ret_dict['pose']
        if args.dataset == '':
            pose -= pose[:, [0]]
            gt_pose -= gt_pose[:, [0]]

        mpjpe += (pose - gt_pose.to(pose.device)).norm(dim=-1).mean()
        err_pa = reconstruction_error(pose.cpu().numpy(), gt_pose.cpu().numpy(), reduction=None)
        pa_mpjpe += np.mean(err_pa[err_pa < 10])
        number += 1
        

        joint_errors = (pose - gt_pose.to(pose.device)).norm(dim=-1)
        correct_joints = (joint_errors < 0.1).float()
        joint_mAP += correct_joints.mean(dim=0)


        joint_mpjpe += joint_errors.mean(dim=0)
        


        

    mpjpe = mpjpe.item() / number
    pa_mpjpe = pa_mpjpe.item() / number
    fps = number*bs  / total_inference_time

    joint_mAP /= number
    joint_mpjpe /= number
    total_mAP = joint_mAP.mean()
    total_mpjpe = joint_mpjpe.mean()


    print(f'Total Params: {params:.2f}M | Inference Speed: {fps:.2f} FPS')
    print(f'Average mAP: {total_mAP * 100:.2f}%')
    print(f'Average MPJPE: {total_mpjpe:.4f}m')
    
    for i, (joint_map, joint_mpjpe_value) in enumerate(zip(joint_mAP, joint_mpjpe)):
        print(f'Joint {i+1} mAP: {joint_map * 100:.2f}% | MPJPE: {joint_mpjpe_value:.4f}m')
    return mpjpe, pa_mpjpe





args = parse_args()
setup_seed(10)
dataset_task = args.dataset
model_type = ''
bs = 32

if dataset_task == 'sloper4d':
    scene_train = [
            'seq002_football_001',
            'seq003_street_002',
            'seq005_library_002',
            'seq008_running_001',
            'seq009_running_002'
        ]
    scene_test = ['seq007_garden_001']
    dataset_root = '/Extra/sloper4d/'
    train_dataset = SLOPER4D_Dataset(dataset_root, scene_train, is_train = True, dataset_path = './Extra/sloper4d/',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, return_smpl = True, augmentation = True, interval = 1)
    test_dataset = SLOPER4D_Dataset(dataset_root, scene_test, is_train = False, dataset_path = './Extra/sloper4d/',
                                return_torch=False, device = 'cuda',
                                fix_pts_num=True, return_smpl = True, interval = 1)
    num_keypoints = 15



elif dataset_task == 'humanm3':
    train_dataset = HumanM3_Dataset(is_train = True,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, augmentation = True, interval = 5)
    test_dataset = HumanM3_Dataset(is_train = False,
                                return_torch=True, device = 'cuda',
                                fix_pts_num=True, interval = 5)
    num_keypoints = 15




train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)


model = SarL().cuda()
state_dict = torch.load(args.state_dict)
model.load_state_dict(state_dict['net'])
mpjpe, pa_mpjpe = test(model, test_loader)
print('MPJPE: '+str(mpjpe) + '; PA-MPJPE:' + str(pa_mpjpe) + ';')
