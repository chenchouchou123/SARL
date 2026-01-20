
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
import einops
from timm.models.layers import DropPath
import pointops
import numpy as np
from pointops import offset2batch, batch2offset
import matplotlib.pyplot as plt
import os

class GroupedLinear(nn.Module):
    __constants__ = ['in_features', 'out_features', "groups"]
    in_features: int
    out_features: int
    groups: int
    weight: torch.Tensor
    
    def __init__(self, in_features: int, out_features: int, groups: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GroupedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        assert in_features % groups == 0
        assert out_features % groups == 0
        assert out_features == groups
        self.weight = nn.Parameter(torch.empty((1, in_features), **factory_kwargs))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input * self.weight).reshape(
            list(input.shape[:-1]) + [self.groups, input.shape[-1] // self.groups]).sum(-1)


class PointBatchNorm(nn.Module):
    """Batch Normalization for Point Clouds data"""
    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError



class GroupedVectorAttention(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        
        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)
        
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias
        if pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        
        self.weight_encoding = nn.Sequential(
            GroupedLinear(embed_channels, groups, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)
    
    def forward(self, feat, coord, reference_index, offset=None):
        query = self.linear_q(feat)
        key = self.linear_k(feat)
        value = self.linear_v(feat)
        
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        
        relation_qk = key - query.unsqueeze(1)
        
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb
        
        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))
        
        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        
        return feat




class DCTFrequencyFilterGVA(nn.Module):

    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 use_spectral_enhance=True,
                 spectral_ratio=0.2,
                 # === 新增的DCT参数 ===
                 lowpass_ratio=0.3,      # 低通滤波器截止比例（前30%为低频）
                 highpass_ratio=0.7,     # 高通开始比例（后30%为高频）
                 low_freq_boost=2.0,     # 低频增强倍数
                 high_freq_suppress=0.1, # 高频抑制倍数
                 learnable_filter=False,
                 debug_visualize=False,      # 是否启用调试可视化
                 save_debug_path="./debug_imgs"):  # 调试图片保存路径): # 是否使用可学习滤波器
        super().__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        self.use_spectral_enhance = use_spectral_enhance
        self.spectral_ratio = spectral_ratio
        self.i_per_group = embed_channels // groups
        
        # DCT参数
        self.lowpass_ratio = lowpass_ratio
        self.highpass_ratio = highpass_ratio
        self.low_freq_boost = low_freq_boost
        self.high_freq_suppress = high_freq_suppress
        self.learnable_filter = learnable_filter
        
        # === 调试参数 ===
        self.debug_visualize = debug_visualize
        self.save_debug_path = save_debug_path
        self.debug_counter = 0  # 计数器，避免生成太多图片
        
        # 创建调试目录
        if debug_visualize and save_debug_path:
            os.makedirs(save_debug_path, exist_ok=True)

        # ============ 空域分支（标准GVA，主干）============
        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias
        if pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )

        self.weight_encoding = nn.Sequential(
            GroupedLinear(embed_channels, groups, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

        # ============ DCT频率滤波分支 ============
        if use_spectral_enhance:
            # === 创新1：构建DCT变换矩阵 ===
            self.register_buffer('dct_matrix', self._build_dct_matrix(self.i_per_group))
            self.register_buffer('idct_matrix', self.dct_matrix.T)  # 逆DCT = DCT转置
            
            # === 创新2：设计频率滤波器 ===
            # 固定的频率响应曲线（非可学习）
            self.register_buffer('frequency_filter', 
                                self._design_frequency_filter(self.i_per_group))
            
            # === 创新3：可选的可学习调制 ===
            if learnable_filter:
                # 可学习的频率调制（但有约束）
                self.learnable_freq_modulation = nn.Parameter(
                    torch.ones(groups, self.i_per_group) * 0.1  # 初始化为小的调制
                )
            
            # === 创新4：自适应门控 ===
            self.spectral_gate = nn.Sequential(
                nn.Linear(embed_channels, embed_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels // 4, 1),
                nn.Sigmoid()
            )
    
    def _build_dct_matrix(self, N):
        """
        构建DCT-II变换矩阵
        
        DCT-II是最常用的DCT变体，OpenCV等库默认使用
        
        公式：
        C(k) = sqrt(2/N) * sum(x(n) * cos(π*k*(2n+1)/(2N)))
        其中 k=0,1,...,N-1, n=0,1,...,N-1
        """
        dct_matrix = torch.zeros(N, N)
        
        for k in range(N):
            for n in range(N):
                if k == 0:
                    # DC component
                    dct_matrix[k, n] = np.sqrt(1.0 / N)
                else:
                    dct_matrix[k, n] = np.sqrt(2.0 / N) * np.cos(
                        np.pi * k * (2 * n + 1) / (2 * N)
                    )
        
        return dct_matrix
    
    def _design_frequency_filter(self, N):
        """
        设计明确的频率滤波器
        
        策略：
        - 低频（0 ~ lowpass_ratio*N）：增强 (x low_freq_boost)
        - 中频（lowpass_ratio*N ~ highpass_ratio*N）：保持 (x 1.0)
        - 高频（highpass_ratio*N ~ N）：抑制 (x high_freq_suppress)
        
        形状：平滑过渡，避免振铃效应
        """
        filter_response = torch.ones(N)
        
        # 计算频带边界
        low_cutoff = int(self.lowpass_ratio * N)
        high_cutoff = int(self.highpass_ratio * N)
        
        # === 低频增强区域 ===
        filter_response[:low_cutoff] = self.low_freq_boost
        
        # === 中频过渡区域（低频→中频）===
        if low_cutoff < high_cutoff:
            # 从low_freq_boost平滑过渡到1.0
            transition_len = high_cutoff - low_cutoff
            transition = torch.linspace(self.low_freq_boost, 1.0, transition_len)
            # 使用余弦窗口平滑过渡
            cosine_window = 0.5 * (1 + torch.cos(torch.linspace(0, np.pi, transition_len)))
            smooth_transition = self.low_freq_boost * cosine_window + 1.0 * (1 - cosine_window)
            filter_response[low_cutoff:high_cutoff] = smooth_transition
        
        # === 高频抑制区域（中频→高频）===
        if high_cutoff < N:
            # 从1.0平滑过渡到high_freq_suppress
            transition_len = N - high_cutoff
            # 使用余弦窗口平滑过渡到抑制
            cosine_window = 0.5 * (1 + torch.cos(torch.linspace(0, np.pi, transition_len)))
            smooth_transition = 1.0 * cosine_window + self.high_freq_suppress * (1 - cosine_window)
            filter_response[high_cutoff:] = smooth_transition
        
        return filter_response
    

    
    def dct_frequency_filtering(self, feat):

        if not self.use_spectral_enhance:
            return feat
        
        if self.debug_visualize:
            self._debug_visualize_ultra_dense_spectrum_final(feat, "before_filtering")
        
        # [N, C] -> [N, g, i]
        grouped = einops.rearrange(feat, "n (g i) -> n g i", g=self.groups)
        N, g, i = grouped.shape
        
        # === Step 1: DCT前向变换 ===
        # grouped: [N, g, i], dct_matrix: [i, i]
        # 对最后一个维度做DCT
        dct_coeffs = torch.matmul(grouped, self.dct_matrix.T)  # [N, g, i]
        
        # === Step 2: 频率滤波 ===
        # 应用设计的滤波器
        filtered_coeffs = dct_coeffs * self.frequency_filter.unsqueeze(0).unsqueeze(0)  # [N, g, i]
        
        # === Step 3: 可选的可学习调制 ===
        if self.learnable_filter:
            # 添加小的可学习调制（但仍然以固定滤波器为主）
            learnable_modulation = torch.tanh(self.learnable_freq_modulation) * 0.2  # 限制在±0.2
            filtered_coeffs = filtered_coeffs * (1 + learnable_modulation.unsqueeze(0))
        
        # === Step 4: DCT逆变换 ===
        enhanced_grouped = torch.matmul(filtered_coeffs, self.dct_matrix)  # [N, g, i]
        
        # === Step 5: 重新组合 ===
        enhanced = einops.rearrange(enhanced_grouped, "n g i -> n (g i)")
        
        if self.debug_visualize:
            self._debug_visualize_ultra_dense_spectrum_final(enhanced, "after_filtering")
            

        return enhanced
    
    def forward(self, feat, coord, reference_index, offset=None):
        N, C = feat.shape

        # ============ 标准空域GVA（主干）============
        query = self.linear_q(feat)
        key = self.linear_k(feat)
        value = self.linear_v(feat)

        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]

        relation_qk = key - query.unsqueeze(1)

        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1).float()
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)

        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        spatial_feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        spatial_feat = einops.rearrange(spatial_feat, "n g i -> n (g i)")

        # ============ DCT频率滤波增强（辅助）============
        if self.use_spectral_enhance:
            enhanced_feat = self.dct_frequency_filtering(spatial_feat)
            
            # 自适应门控：根据特征复杂度决定滤波强度
            gate = self.spectral_gate(spatial_feat) * self.spectral_ratio   # [N, 1]
            
            # 融合：保持空域为主，频域为辅
            output = (1 - gate) * spatial_feat + gate * enhanced_feat
        else:
            output = spatial_feat

        return output



# ============ Block（简化）============
class Block(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False,
                 use_gcn_enhance=False):
        super(Block, self).__init__()
        
        # 根据use_gcn_enhance选择注意力模块
        if use_gcn_enhance:
            self.attn = DCTFrequencyFilterGVA(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                use_spectral_enhance=True,
                spectral_ratio=0.1,          
                lowpass_ratio=0.4,           
                highpass_ratio=0.7,          
                low_freq_boost=1.5,          
                high_freq_suppress=0.3,      
                learnable_filter=True       
            )

        else:
            self.attn = GroupedVectorAttention(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias
            )
        
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    
    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index, offset) \
            if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index, offset)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


# ============ BlockSequence（简化）============
class BlockSequence(nn.Module):
    def __init__(self,
                 depth,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False,
                 use_gcn_enhance=False):
        super(BlockSequence, self).__init__()
        
        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0. for _ in range(depth)]
        
        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
                use_gcn_enhance=use_gcn_enhance
            )
            self.blocks.append(block)
    
    def forward(self, points):
        coord, feat, offset = points
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


# ============ GridPool、UnpoolWithSkip（保持不变）============
class GridPool(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                            reduce="min") if start is None else start
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


class UnpoolWithSkip(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bias=True, skip=True, backend="map"):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]
        
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = pointops.interpolation(coord, skip_coord, self.proj(feat), offset, skip_offset)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


# ============ Encoder（简化 - 去除GCN路径）============
class Encoder(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 grid_size=None,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 use_gcn_enhance=False):
        super(Encoder, self).__init__()
        
        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )
        
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint,
            use_gcn_enhance=use_gcn_enhance
        )
    
    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster


# ============ Decoder（简化）============
class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 embed_channels,
                 groups,
                 depth,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 unpool_backend="map",
                 use_gcn_enhance=False):
        super(Decoder, self).__init__()
        
        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend
        )
        
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint,
            use_gcn_enhance=use_gcn_enhance
        )
    
    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False,
                 use_gcn_enhance=False):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
            use_gcn_enhance=use_gcn_enhance
        )
    
    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


# ============ 主模型（简化 - 去除GCN路径）============
class FEPT(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 patch_embed_depth=1,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=8,
                 enc_depths=(2, 2, 6, 2),
                 enc_channels=(96, 192, 384, 512),
                 enc_groups=(12, 24, 48, 64),
                 enc_neighbours=(16, 16, 16, 16),
                 dec_depths=(1, 1, 1, 1),
                 dec_channels=(48, 96, 192, 384),
                 dec_groups=(6, 12, 24, 48),
                 dec_neighbours=(16, 16, 16, 16),
                 grid_sizes=(0.06, 0.12, 0.24, 0.48),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0,
                 enable_checkpoint=False,
                 unpool_backend="map",
                 use_gcn_enhance=False):  # 只保留这一个开关
        super(FEPT, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        
        # 断言检查
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        
        # Patch Embed
        self.patch_embed = GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
            use_gcn_enhance=use_gcn_enhance
        )
        
        # Drop path rates
        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        
        # Encoder and Decoder stages
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                use_gcn_enhance=use_gcn_enhance
            )
            
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
                use_gcn_enhance=use_gcn_enhance
            )
            
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            PointBatchNorm(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], num_classes)
        ) if num_classes > 0 else nn.Identity()
    
    def forward(self, points):
        # [B, 3, N]
        points = points.permute(0, 2, 1)
        B, N, C = points.shape
        coord = points.reshape([-1, C])
        feat = points.reshape([-1, C])
        offset = ((torch.arange(B, device=feat.device) + 1) * N).int()
        
        # Forward pass
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)
            skips.append([points])
        
        points = skips.pop(-1)[0]
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
        
        coord, feat, offset = points
        seg_logits = self.seg_head(feat)
        return seg_logits, feat













