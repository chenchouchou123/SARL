import torch
import torch.nn as nn
from openpoints.models.PCM.PCM_utils import knn_point,index_points
from openpoints.models.layers import furthest_point_sample
import torch.nn.functional as F



class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBNReLU1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GGA(nn.Module):
    def __init__(self, in_channels, out_channels, S=1024, kneighbors=32, k_stride=1, local_feat_dim=64):
        super(GGA, self).__init__()
        # Removed self.embedding to move feature extraction after transformation
        self.combine_pos = True
        self.S = S
        self.kneighbors = kneighbors
        self.k_stride = k_stride
        self.lrf_bias = nn.Parameter(torch.eye(3))  # 可学习的 LRF 偏置
        self.normalize = "center"
        
        # Adjust input dim for local extractor: 3 (lrf_points) + feat_dim (grouped feat - center feat)

        local_input_dim = 3 + in_channels  # lrf_points (3) + (grouped_points - new_points) (feat_dim)
        
        # MLP for local feature extraction on concatenated local coords and feats
        self.local_feat_extractor = nn.Sequential(
            nn.Linear(local_input_dim, local_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(local_feat_dim, local_feat_dim),
            nn.ReLU(inplace=True)
        )
        self.aggregate = nn.Linear(local_feat_dim, out_channels)  # Aggregate to output dim
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, 27]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, 27]))

    def forward(self, p, x=None):
        if isinstance(p, dict):
            p, x = p['pos'], p.get('x', None)
        
        # Prepare input features
        if x is None:
            feat = p.transpose(1, 2).contiguous()  # [B, 3, N]
        else:
            if self.combine_pos:
                feat = torch.cat([x, p.transpose(1, 2)], dim=1).contiguous()  # [B, in_channels + 3, N]
            else:
                feat = x  # [B, in_channels, N]
        
        points = feat.transpose(1, 2).contiguous()  # [B, N, D], where D = feat channels
        
        # Ensure points is tensor
        if not isinstance(points, torch.Tensor):
            raise TypeError(f"Expected points to be torch.Tensor, got {type(points)}")
        
        # Compute LRF transformed points and grouped features
        GEL_points, new_xyz, grouped_points, new_points = (process_point_cloud_to_GEL





            (
            xyz=p,
            points=points,
            S=self.S,
            kneighbors=self.kneighbors,
            k_stride=self.k_stride,
            lrf_bias=self.lrf_bias,
            normalize=self.normalize,
            training=self.training,
            affine_alpha=self.affine_alpha,
            affine_beta=self.affine_beta
        ))  # lrf_points: [B, S, K', 3], new_xyz: [B, S, 3], grouped_points: [B, S, K', D], new_points: [B, S, D]
        
        # Prepare local input: cat(lrf_points, grouped_points - new_points.unsqueeze(2))
        center_points = new_points.unsqueeze(2)  # [B, S, 1, D]
        local_input = torch.cat([GEL_points, grouped_points - center_points], dim=-1)  # [B, S, K', 3 + D]
        
        # Reshape for point-wise MLP: [B*S*K', 3 + D]
        B, S, K, _ = local_input.shape
        local_feats = local_input.view(B * S * K, -1)
        
        # Apply feature extractor MLP (this is the feature extraction after transformation)
        local_feats = self.local_feat_extractor(local_feats)  # [B*S*K', local_feat_dim]
        
        # Reshape back: [B, S, K', local_feat_dim]
        local_feats = local_feats.view(B, S, K, -1)
        
        # Aggregate (max pooling over neighbors)
        aggregated_feats = torch.max(local_feats, dim=2)[0]  # [B, S, local_feat_dim]
        
        # Project to output channels
        output_feats = self.aggregate(aggregated_feats)  # [B, S, out_channels]
        
        # Return
        return GEL_points, output_feats, new_xyz

def process_point_cloud_to_GEL(xyz, points, S, kneighbors, k_stride, lrf_bias, normalize=None, training=True, affine_alpha=None, affine_beta=None):
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Expected points to be torch.Tensor, got {type(points)}")
    B, N, _ = xyz.shape
    # Point sampling
    if S == N:
        new_xyz = xyz  # [B, S, 3]
        new_points = points  # [B, S, D]
    else:
        fps_idx = furthest_point_sample(xyz, S).long()  # [B, S]
        fps_idx = torch.sort(fps_idx, dim=-1)[0]
        new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]
        new_points = index_points(points, fps_idx)  # [B, S, D]
    # KNN neighbor search
    idx = knn_point(kneighbors, xyz, new_xyz, training=training)  # [B, S, K]
    idx = idx[:, :, ::k_stride]  # [B, S, K']
    # Group points
    grouped_xyz = index_points(xyz, idx)  # [B, S, K', 3]
    grouped_points = index_points(points, idx)  # [B, S, K', D]
    # Compute LRF transformed points
    lrf_points = compute_GEL(new_xyz, grouped_xyz, lrf_bias)  # [B, S, K', 3]
    return lrf_points, new_xyz, grouped_points, new_points

def compute_GEL(points, neighbors, lrf_bias):
    B, N, K, _ = neighbors.shape
    diff = neighbors - points.unsqueeze(2)
    cov_matrix = torch.matmul(diff.permute(0, 1, 3, 2), diff)
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    normal = eigvecs[..., 0]
    t1 = eigvecs[..., 1]
    t2 = torch.cross(normal, t1, dim=-1)
    lrf_matrix = torch.stack([t1, t2, normal], dim=-1)
    adjusted_lrf_matrix = torch.matmul(lrf_matrix, lrf_bias)
    lrf_points = torch.matmul(diff, adjusted_lrf_matrix)
    return lrf_points


    

    
    
class GGA_EXTRACTION(nn.Module):

    def __init__(self, in_channels, out_channels, S=1024, kneighbors=32, 
                 k_stride=1, local_feat_dim=64):
        super(GGA_EXTRACTION, self).__init__()
        
        self.combine_pos = True
        self.S = S
        self.kneighbors = kneighbors
        self.k_stride = k_stride
        self.local_feat_dim = local_feat_dim
        
        self.lrf_bias = nn.Parameter(torch.eye(3))
        self.normalize = "center"
        

        self.coord_embedding = nn.Sequential(
            nn.Linear(3, local_feat_dim // 2),
            nn.BatchNorm1d(local_feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(local_feat_dim // 2, local_feat_dim // 2),
            nn.BatchNorm1d(local_feat_dim // 2),
            nn.ReLU(inplace=True)
        )
        

        self.geometric_encoder = nn.Sequential(
            nn.Linear(3, local_feat_dim // 2),
            nn.BatchNorm1d(local_feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(local_feat_dim // 2, local_feat_dim // 2),
            nn.BatchNorm1d(local_feat_dim // 2),
            nn.ReLU(inplace=True)
        )
        

        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(3, local_feat_dim // 2),
            nn.BatchNorm1d(local_feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(local_feat_dim // 2, local_feat_dim // 2),
            nn.BatchNorm1d(local_feat_dim // 2),
            nn.ReLU(inplace=True)
        )
        

        edge_input_dim = 3 + local_feat_dim // 2 * 3
        self.edge_conv = nn.Sequential(
            nn.Linear(edge_input_dim, local_feat_dim),
            nn.BatchNorm1d(local_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(local_feat_dim, local_feat_dim),
            nn.BatchNorm1d(local_feat_dim),
            nn.ReLU(inplace=True)
        )
        

        self.feature_fusion = nn.Sequential(
            nn.Linear(local_feat_dim * 2, local_feat_dim),
            nn.BatchNorm1d(local_feat_dim),
            nn.ReLU(inplace=True)
        )
        

        self.geo_attention = nn.Sequential(
            nn.Linear(local_feat_dim // 2, local_feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(local_feat_dim // 4, 1),
            nn.Sigmoid()
        )
        

        self.aggregate = nn.Linear(local_feat_dim, out_channels)
        
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, 27]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, 27]))
    
    def forward(self, p, x=None):

        if isinstance(p, dict):
            p = p['pos']
        
        points = p  # [B, N, 3]
        

        lrf_points, new_xyz, grouped_xyz_raw, new_points = process_point_cloud_to_lrf(
            xyz=p,
            points=points,
            S=self.S,
            kneighbors=self.kneighbors,
            k_stride=self.k_stride,
            lrf_bias=self.lrf_bias,
            normalize=self.normalize,
            training=self.training,
            affine_alpha=self.affine_alpha,
            affine_beta=self.affine_beta
        )
        
        B, S, K, _ = lrf_points.shape
        

        # 中心点特征
        center_coords = new_points.view(B * S, 3)
        center_embedding = self.coord_embedding(center_coords)  # [B*S, D//2]
        center_embedding = center_embedding.unsqueeze(1).expand(-1, K, -1)
        center_embedding = center_embedding.reshape(B * S * K, -1)  # [B*S*K, D//2]
        
        # 邻域点特征
        neighbor_coords = grouped_xyz_raw.view(B * S * K, 3)
        neighbor_embedding = self.coord_embedding(neighbor_coords)  # [B*S*K, D//2]
        

        lrf_flat = lrf_points.view(B * S * K, 3)
        geo_feat = self.geometric_encoder(lrf_flat)  # [B*S*K, D//2]
        

        center_coords_expanded = new_points.unsqueeze(2).expand(-1, -1, K, -1)
        relative_pos = grouped_xyz_raw - center_coords_expanded
        relative_pos_flat = relative_pos.view(B * S * K, 3)
        relative_feat = self.relative_pos_encoder(relative_pos_flat)  # [B*S*K, D//2]
        

        edge_input = torch.cat([
            lrf_flat,              # [B*S*K, 3]
            center_embedding,      # [B*S*K, D//2]
            neighbor_embedding,    # [B*S*K, D//2]
            relative_feat          # [B*S*K, D//2]
        ], dim=-1)  # [B*S*K, 3 + D//2*3]
        
        edge_conv_feat = self.edge_conv(edge_input)  # [B*S*K, D]
        

        combined = torch.cat([
            geo_feat,              # [B*S*K, D//2] 几何特征
            neighbor_embedding,    # [B*S*K, D//2] 坐标嵌入
            edge_conv_feat         # [B*S*K, D] 边特征
        ], dim=-1)  # [B*S*K, 2D]
        
        fused_feat = self.feature_fusion(combined)  # [B*S*K, D]
        fused_feat = fused_feat.view(B, S, K, -1)
        

        geo_feat_reshaped = geo_feat.view(B, S, K, -1)
        

        geo_weights = self.geo_attention(
            geo_feat_reshaped.view(B * S * K, -1)
        ).view(B, S, K, 1)
        

        geo_weights = torch.softmax(geo_weights, dim=2)
        

        aggregated_feats = (fused_feat * geo_weights).sum(dim=2)  # [B, S, D]
        

        output_feats = self.aggregate(aggregated_feats)  # [B, S, out_channels]
        
        return lrf_points, output_feats, new_xyz