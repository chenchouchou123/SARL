import torch
import torch.nn as nn
import numpy as np
from .FEPT import FEPT
from models.lpformer.lpformer import BertSelfAttention
from .kpt_config import kps15_flip_pairs, kps15_bone, kps15_bone_ratio
import torch.nn.functional as F
from .GGA import GGA_EXTRACTION

class Segmentation(nn.Module):
    def __init__(self, input_dim = 256, part_num = 16):
        super().__init__()

        self.part_num = part_num
        '''
        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        '''
        self.convs1 = nn.Conv1d(input_dim * 3, input_dim * 2, 1)
        self.convs2 = nn.Conv1d(input_dim * 2, input_dim, 1)
        self.convs3 = nn.Conv1d(input_dim, self.part_num, 1)

        self.bns1 = nn.BatchNorm1d(input_dim * 2)
        self.bns2 = nn.BatchNorm1d(input_dim)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean):
        batch_size, _, N = x.size()

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        #cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        #cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature], dim=1)  # 1024 * 3 + 64

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x

class NormalEstimation(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 3, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean):
        N = x.size(2)

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)
        
        x = torch.cat([x_max_feature, x_mean_feature, x], dim=1)

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x

class Regression(nn.Module):
    def __init__(self, input_dim = 256, out_dim = 16): #原input_dim = 256
        super().__init__()

        '''
        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        '''
        self.convs1 = nn.Conv1d(input_dim * 3, input_dim * 2, 1)
        self.convs2 = nn.Conv1d(input_dim * 2, input_dim, 1)
        self.convs3 = nn.Conv1d(input_dim, out_dim, 1)

        self.bns1 = nn.BatchNorm1d(input_dim * 2)
        self.bns2 = nn.BatchNorm1d(input_dim)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean):
        batch_size, _, N = x.size()

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        #cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        #cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature], dim=1)  # 1024 * 3 + 64

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        feat = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(feat)
        # x = x.mean(dim = -1)
        return x, feat



class FeatureFusion(nn.Module):
    def __init__(self, local_dim=24, global_dim=48, out_dim=48):
        super(FeatureFusion, self).__init__()
        self.local_proj = nn.Linear(local_dim, out_dim)
        

        self.cross_attn = nn.Sequential(
            nn.Linear(out_dim + global_dim, out_dim),
            nn.Sigmoid()
        )
        
        self.output_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, local_feat, global_feat):
        # local_feat: (B, N, 24), global_feat: (B, N, 48)
        local_proj = self.local_proj(local_feat)  # (B, N, 48)
        

        attn_input = torch.cat([local_proj, global_feat], dim=-1)  # (B, N, 96)
        attn_weight = self.cross_attn(attn_input)  # (B, N, 48)
        

        fused = attn_weight * local_proj + (1 - attn_weight) * global_feat
        output = self.output_mlp(fused)  # (B, N, 48)
        return output


class SarL(nn.Module):
    # sarl
    def __init__(self, pmg_cfg = None):
        super().__init__()
        self.cfg = pmg_cfg
        pose_dim = self.cfg.NETWORK.pose_dim
        self.pose_dim = pose_dim
        self.encoder = FEPT(3, pose_dim + 1, enc_depths = (2,2,2,2), enc_channels = (16, 32, 64, 128), \
                                         enc_groups = (2, 4, 8, 16),use_gcn_enhance=True)



        # self.encoder = PointTransformerV3(
        #     in_channels=3,  # 保持为3（XYZ），适合LiDAR点云
        #     enc_depths=(2, 2, 2, 6, 2),  # 加深中间层，提升多尺度特征提取（参考官方Base）
        #     enc_channels=(64, 128, 256, 384, 512),  # 增大通道，提高容量（你的原版太小，可能欠拟合）
        #     enc_num_head=(4, 8, 16, 24, 32),  # 头数匹配通道（channel % head == 0，确保可分）
        #     stride=(2, 2, 2, 2),  # 增加一个stride，匹配4个编码阶段（下采样4次）
        #     enc_patch_size=(1024, 1024, 1024, 1024, 1024),  # 保持大patch，适合稀疏点云
        #     dec_depths=(1, 1, 1, 1),  # 解码器浅层（官方建议，每阶段1层，减少计算）
        #     dec_channels=(48, 128, 256, 384),  # 从小到大匹配编码器上采样，最终feat C=64（可根据你的下游任务调整为48）
        #     dec_num_head=(4, 8, 16, 24),  # 头数匹配dec_channels
        #     dec_patch_size=(1024, 1024, 1024, 1024),  # 一致
        #     cls_mode=False,  # 保持禁用，适合回归任务如HPE
        #     drop_path=0.3,  # 增大到0.3（官方推荐），增强正则化，防过拟合
        #     enable_flash=True,  # 如果你的环境支持FlashAttention（CUDA>=11.6），启用以加速；否则设False并减小patch_size到512试调
        #     mlp_ratio=4,  # 添加MLP比率（官方默认），提升前馈层容量
        #     qkv_bias=True  # 启用QKV偏置（官方设置），改善注意力稳定性
        # )

        # pcm
        # self.encoder = PointMambaEncoder(in_channels=3, embed_dim=24, groups=1, res_expansion=1.0,
        #          activation="relu", bias=False, use_xyz=True, normalize="anchor",
        #          dim_expansion=[1, 1, 2, 1], pre_blocks=[1, 1, 1, 1], pos_blocks=[0, 0, 0, 0],
        #          k_neighbors=[12, 12, 12, 12], k_strides=[1, 1, 1, 1], reducers=[1, 1, 1, 1],
        #          mamba_blocks=[1, 2, 2, 4],
        #          mamba_layers_orders=["xyz","xzy","yxz","yzx","zxy","zyx","hilbert","z","z-trans"],
        #          use_order_prompt=True, prompt_num_per_order=6,
        #          rms_norm=True, fused_add_norm=True, residual_in_fp32=True,
        #          bimamba_type="v2", drop_path_rate=0.1,
        #          mamba_pos=True, pos_type='share', pos_proj_type="linear",
        #          grid_size=0.02, combine_pos=False, block_residual=True,
        #          use_windows=False, windows_size=1200,
        #          cls_pooling="max")
        # # #pcm-tiny
        # self.encoder = PointMambaEncoder(in_channels=3, embed_dim=24, groups=1, res_expansion=1.0,
        #          activation="relu", bias=False, use_xyz=True, normalize="anchor",
        #          dim_expansion=[1, 1, 1, 2], pre_blocks=[1, 1, 1, 1], pos_blocks=[0, 0, 0, 0],
        #          k_neighbors=[12, 12, 12, 12], k_strides=[1, 1, 1, 1], reducers=[1, 1, 1, 1],
        #          mamba_blocks=[1, 1, 2, 2],
        #          mamba_layers_orders=["xyz","xzy","yxz","yzx","zxy","zyx"],
        #          use_order_prompt=True, prompt_num_per_order=6,
        #          rms_norm=True, fused_add_norm=True, residual_in_fp32=True,
        #          bimamba_type="v2", drop_path_rate=0.1,
        #          mamba_pos=True, pos_type='share', pos_proj_type="linear",
        #          grid_size=0.02, combine_pos=False, block_residual=True,
        #          use_windows=False, windows_size=1200,
        #          cls_pooling="max")
        dim_mid = 48
        self.to_48_linear = nn.Linear(96, 48)
        self.reg = Regression(dim_mid, 3)
        if self.cfg.NETWORK.VOTE:
            self.seg = Segmentation(dim_mid, pose_dim + 1)
        else:
            self.reg_ = nn.Sequential(
                nn.Linear(dim_mid, 24),
                nn.Linear(24, self.pose_dim * 3)
            )
        self.bone_seg = False
        self.criterion_segment = torch.nn.CrossEntropyLoss()
        

        self.gga = GGA_EXTRACTION(
                    in_channels=3,
                    out_channels=24,
                    S=1024,
                    kneighbors=32
                )
        self.feature_fusion = FeatureFusion(local_dim=24, global_dim=48, out_dim=48)
        
        if self.cfg.NETWORK.REFINE:
            self.kpt_embedding = nn.Embedding(pose_dim, 32)
            self.kpt_linear = nn.Linear(32 + 3, dim_mid)
            self.KPTR = nn.Sequential(
                BertSelfAttention(dim_mid + 3, 3),
                BertSelfAttention(dim_mid + 3, 3)
            )
            self.mlp_xyz = nn.Sequential(
                nn.Linear(dim_mid + 3, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.Linear(8, 3),
            )

    def forward(self, xyz):
        B,N,_ = xyz.shape

        n_point,local_feature,center_p = self.gga(xyz)
        seg, feat = self.encoder(xyz.permute(0,2,1))   #加入mamba encoder前ptv2
        feat = feat.view(B,N,-1)

        

        
        feat = self.feature_fusion(local_feature,feat)
        feat = feat.permute(0,2,1)


        
        x_max = torch.max(feat, dim=-1)[0]  
        x_mean = torch.mean(feat, dim=-1)
        x_r, feat_r = self.reg(feat, x_max, x_mean)
        


        
        if self.cfg.NETWORK.VOTE:
            x_s = self.seg(feat, x_max, x_mean)
            ret_dict = {'seg':x_s, 'reg':x_r, 'xyz':xyz.permute(0,2,1)}
            r_dict = self.soft_reg_pose(ret_dict)
            ret_dict.update(r_dict)
        else:
            r_dict, ret_dict = {}, {}
            feat_r = feat_r.mean(dim = -1)
            # import pdb; pdb.set_trace()
            r_dict['pose_r'] = self.reg_(feat_r).reshape(B,self.pose_dim,3)
            ret_dict.update(r_dict)
        feat = feat.permute(0,2,1) #[B,N,C]                                        
        if self.cfg.NETWORK.REFINE:
            pose = r_dict['pose_r']
            xyz_feat = torch.cat([xyz, feat], dim = -1) #[B,N,C+3]
            kpt_embedding = self.kpt_embedding(torch.arange(self.pose_dim).to(xyz.device)).unsqueeze(0).repeat(B, 1, 1)
            kpt_feat = self.kpt_linear(torch.cat([pose, kpt_embedding], dim = -1))
            kpt_feat = torch.cat([pose, kpt_feat], dim = -1)
            all_input_feat = torch.cat([kpt_feat, xyz_feat], dim = 1)
            all_output = self.KPTR(all_input_feat)
            all_pose = self.mlp_xyz(all_output)[:,:self.pose_dim,:3]
            ret_dict.update({'pose':all_pose, 'feat':feat, 'skeleton_feat': all_output[:,:self.pose_dim,3:]})
            # ret_dict.update({'gal_loss':gal_loss})
        else:
            ret_dict.update({'pose':ret_dict['pose_r'], 'feat':feat})
        
        return ret_dict

    def soft_reg_pose(self, input_):
        if isinstance(input_,dict):
            ret_dict = input_
            xyz = ret_dict['xyz'].permute(0,2,1)
        else:
            xyz = input_.float().cuda()
            ret_dict = self.forward(xyz.permute(0,2,1))
        batch_size = xyz.shape[0]
        class_ = ret_dict['seg'].permute(0,2,1)
        class_ = torch.softmax(class_, dim = 2)
        max_class, _ = torch.max(class_, dim = 2)
        class_mask = (class_[:,:,-1] < 0.6) & (max_class > 0.6)
        regress_ = ret_dict['reg'].permute(0,2,1)
        reg_pose = regress_ + xyz
        pose_r = torch.zeros([batch_size, self.pose_dim, 3]).to(class_.device)
        class_sum_all = []
        for b in range(batch_size):
            class_this = class_[b, class_mask[b]]#[N,15]
            class_sum_this = torch.sum(class_this, dim = 0, keepdim = True)#.unsqueeze(0)
            class_sum_all.append(class_sum_this)
            class_k = class_this / (class_sum_this + 1e-8)
            pose = torch.mm(class_k.permute(1,0), reg_pose[b, class_mask[b], :])
            pose_r[b] = pose[:self.pose_dim,:]
        class_sum_all = torch.cat(class_sum_all, dim = 0) #[B,15]
        class_sum_all = class_sum_all / class_sum_all.sum(dim = 1, keepdim = True)
        r_dict = {}
        r_dict.update({'pose_r':pose_r, 'class_sum':class_sum_all, 'reg':regress_})
        return r_dict

    def reg_pose(self, input_):
        if isinstance(input_,dict):
            ret_dict = input_
            xyz = ret_dict['xyz'].permute(0,2,1)
        else:
            xyz = input_.float().cuda()
            ret_dict = self.forward(xyz.permute(0,2,1))
        class_ = ret_dict['seg'].permute(0,2,1)
        class_ = torch.softmax(class_, dim = 2)
        max_class, _ = torch.max(class_, dim = 2)
        ma_c = torch.argmax(class_, dim = 2)
        class_mask = (class_[:,:,-1] < 0.8) & (max_class > 0.6)
        regress_ = ret_dict['reg'].permute(0,2,1)
        reg_pose = regress_ + xyz
        pose_r = torch.zeros([class_.shape[0], 15, 3]).to(class_.device)
        
        for b in range(class_.shape[0]):
            class_this = class_[b, class_mask[b]]#[N,15]
            #print(class_this.shape)
            class_sum_this = torch.sum(class_this, dim = 0).unsqueeze(0)
            class_k = class_this / (class_sum_this + 1e-8)
            reg_pose_this = reg_pose[b,class_mask[b]]
            #print(reg_pose.shape, class_k.shape)
            pose = torch.mm(class_k.permute(1,0), reg_pose[b, class_mask[b], :])
            pose_r[b] = pose[:15,:]
            #pose_r[b] -= pose_r[b,2,:].unsqueeze(0)
        
        r_dict = {}
        r_dict.update({'reg':pose_r, 'seg':ma_c})
        return r_dict

    def keypoint_3d_loss(self, criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss if 3D keypoint annotations are available.
        """
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            # import pdb; pdb.set_trace()
            # return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            return (pred_keypoints_3d - gt_keypoints_3d).norm(dim = -1).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).cuda()

    def seg_loss(self, pred_seg, seg_label):
        # print(pred_seg.max(dim = 1)[0].mean(dim = 1))
        return self.criterion_segment(pred_seg, seg_label)

    def get_reg_loss(self, ret_dict, sample):
        pred_reg = ret_dict['reg']
        class_label = sample['seg_label'].to(pred_reg.device)
        pose = sample['smpl_joints24_local'].float().to(pred_reg.device) if self.pose_dim == 24 else sample['smpl_joints_local'].float().to(pred_reg.device)
        pcd = sample['human_points_local'].float().to(pred_reg.device)
        reg_label = torch.zeros_like(pred_reg).to(pred_reg.device)
        class_label_all = class_label < self.pose_dim
        
        for b in range(reg_label.shape[0]):
            class_label_this = class_label_all[b]
            # import pdb; pdb.set_trace()
            reg_label[b,class_label_this,:] = pose[b, class_label[b, class_label_this], :] - pcd[b, class_label_this, :]
        
        loss = reg_label[class_label_all] - pred_reg[class_label_all]
        if loss.shape[0] > 0:
            loss = loss.norm(dim = -1).mean()
        else:
            loss = torch.tensor(0.0)
        return loss

    def prior_loss(self, ret_dict, sample):
        f_pairs = np.array(kps15_flip_pairs)
        bone_index = np.array(kps15_bone)
        kpts_bone_ratio = np.array(kps15_bone_ratio)
        pose = ret_dict['pose']
        gt_pose = sample['smpl_joints24_local'] if self.pose_dim == 24 else sample['smpl_joints_local']

        kpts_bone_0, kpts_bone_1 = pose[:,bone_index[:,0]], pose[:,bone_index[:,1]]
        kpts_bone = kpts_bone_1 - kpts_bone_0
        kpts_bone_leng = kpts_bone.norm(dim = -1) #[B,N]

        gt_bone_0, gt_bone_1 = gt_pose[:,bone_index[:,0]], gt_pose[:,bone_index[:,1]]
        gt_bone = gt_bone_0 - gt_bone_1
        gt_bone_leng = gt_bone.norm(dim = -1)
        # import pdb; pdb.set_trace()
        pair_0, pari_1 = kpts_bone_leng[:,f_pairs[:,0]], kpts_bone_leng[:,f_pairs[:,1]] #[B,N]
        pair_loss = (pair_0 - pari_1).abs().mean()
        bone_ratio_diff = kpts_bone_leng[:,kpts_bone_ratio[:,0]] / (kpts_bone_leng[:,kpts_bone_ratio[:,1]] + 1e-8) \
            - gt_bone_leng[:,kpts_bone_ratio[:,0]] / (gt_bone_leng[:,kpts_bone_ratio[:,1]] + 1e-8)
        bone_ratio_loss = bone_ratio_diff.abs().mean() 
        return bone_ratio_loss + pair_loss

    def all_loss(self, ret_dict, sample):
        lambda_gal = 1e4
        if self.pose_dim == 24:
            # sample['smpl_joints_local'] = sample['smpl_joints24_local']
            gt_joints = sample['smpl_joints24_local']
        else:
            gt_joints = sample['smpl_joints_local']
        joints_loss = (ret_dict['pose'] - gt_joints).norm(dim = -1).mean()
        seg_loss = torch.tensor(0.0).to(joints_loss.device)
        reg_loss = torch.tensor(0.0).to(joints_loss.device)
        prior_loss = torch.tensor(0.0).to(joints_loss.device)
        # gal_loss = ret_dict['gal_loss']*lambda_gal
        if 'pose_r' in ret_dict.keys():
            joints_loss += (ret_dict['pose_r'] - gt_joints).norm(dim = -1).mean()
        
        if 'seg' in ret_dict.keys() and 'reg' in ret_dict.keys() and self.cfg.LOSS.VOTE_LOSS:
            seg_loss += self.seg_loss(ret_dict['seg'], sample['seg_label'].to(ret_dict['seg'].device))
            reg_loss += self.get_reg_loss(ret_dict, sample)
        if self.cfg.LOSS.PRIOR_LOSS:
            prior_loss += self.prior_loss(ret_dict, sample)
        # all_loss = joints_loss + seg_loss + reg_loss #+ prior_loss
        all_loss = joints_loss + seg_loss + reg_loss
        # all_loss = joints_loss + seg_loss + reg_loss + prior_loss
        loss_dict = {'loss':all_loss, 'loss_joint':joints_loss, 'loss_seg':seg_loss, 'loss_prior':prior_loss}
        # print(loss_dict, (ret_dict['pose'] - gt_joints).norm(dim = -1).mean())
        return loss_dict
