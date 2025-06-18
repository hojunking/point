import sys
import torch.nn as nn
import spconv.pytorch as spconv

try:
    import ocnn
except ImportError:
    ocnn = None

from collections import OrderedDict
from pointcept.models.utils.structure import Point
from pointcept.engines.hooks import HookBase


def is_ocnn_module(module):
    if ocnn is not None:
        ocnn_modules = (
            ocnn.nn.OctreeConv,
            ocnn.nn.OctreeDeconv,
            ocnn.nn.OctreeGroupConv,
            ocnn.nn.OctreeDWConv,
        )
        return isinstance(module, ocnn_modules)
    else:
        return False


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            elif is_ocnn_module(module):
                if isinstance(input, Point):
                    input.octree.features[-1] = module(
                        input.feat[input.octree_order], input.octree, input.octree.depth
                    )
                    input.feat = input.octree.features[-1][input.octree_inverse]
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PointModel(PointModule, HookBase):
    r"""PointModel
    placeholder, PointModel can be customized as a Pointcept hook.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.layers import DropPath # PointTransformerV3에서 DropPath가 이미 사용될 수 있으므로, 필요 없으면 제거

# Note: @MODULES.register_module() 데코레이터를 사용하지 않음
class BFABlock(PointModule): # PointModule을 상속받아 PointSequential 내에서 사용 가능하게 함
    def __init__(
        self,
        in_channels: int, # PTv3 디코더 최종 출력 채널 (Config 상 64 예상)
        semantic_out_channels: int, # 강화된 Semantic Feature 출력 채널 (Config 상 64 예상)
        boundary_feature_channels: int = 128, # BFANet 논문의 classifier_feat/margin_feat 출력 채널 (128)
        num_heads: int = 8, # Attention head 수 (BFANet 논문 참고)
        dropout: list = [0.0, 0.0], # BFANet 논문 코드의 dropout 값
    ):
        super().__init__()
        self.in_channels = in_channels
        self.semantic_out_channels = semantic_out_channels
        self.boundary_feature_channels = boundary_feature_channels
        self.num_heads = num_heads
        # BFANet 코드에서 C=128, H=8이므로 C_prime = C/H = 16
        self.head_dim = self.boundary_feature_channels // self.num_heads 
        self.scale = 1.0 # BFANet 코드와 동일

        # Corresponding to classifier_feat and margin_feat in BFANet_SegHeader
        self.semantic_init_mlp = nn.Sequential(
            nn.Linear(in_channels, boundary_feature_channels), # in_channels (64) -> 128
            nn.BatchNorm1d(boundary_feature_channels),
            nn.LeakyReLU()
        )
        self.boundary_init_mlp = nn.Sequential(
            nn.Linear(in_channels, boundary_feature_channels), # in_channels (64) -> 128
            nn.BatchNorm1d(boundary_feature_channels),
            nn.LeakyReLU()
        )

        # Corresponding to sem_qkv and margin_qkv in BFANet_SegHeader
        self.sem_qkv = nn.Sequential(
            nn.Linear(boundary_feature_channels, boundary_feature_channels * 3), # 128 -> 384 (Q,K,V 각 128)
            nn.BatchNorm1d(boundary_feature_channels * 3),
            nn.LeakyReLU()
        )
        self.margin_qkv = nn.Sequential(
            nn.Linear(boundary_feature_channels, boundary_feature_channels * 3), # 128 -> 384 (Q,K,V 각 128)
            nn.BatchNorm1d(boundary_feature_channels * 3),
            nn.LeakyReLU()
        )
        
        # Corresponding to fusion_q in BFANet_SegHeader
        # input: (N, 2 * boundary_feature_channels) = (N, 256)
        # output: (N, boundary_feature_channels) = (N, 128)
        self.fusion_q = nn.Sequential(
            nn.Linear(boundary_feature_channels * 2, boundary_feature_channels * 2), # 256 -> 256
            nn.BatchNorm1d(boundary_feature_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(boundary_feature_channels * 2, boundary_feature_channels), # 256 -> 128
            nn.BatchNorm1d(boundary_feature_channels),
            nn.LeakyReLU(),
        )

        self.attn_drop = nn.Dropout(0.0) # BFANet 코드와 동일
        self.softmax = nn.Softmax(dim=-1)

        # Boundary Prediction Head
        # Takes `margin_out_fused` (N, 128) and outputs (N, 1) probability
        self.boundary_pred_head = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(boundary_feature_channels, 64), # 128 -> 64
            nn.BatchNorm1d(64),
            nn.PReLU(), # PReLU Activation from BFANet code
            nn.Dropout(dropout[1]),
            nn.Linear(64, 1, bias=True), # 64 -> 1 (binary classification)
            nn.Sigmoid() # Output as probability
        )

        # Remap semantic feature to desired output channel for DefaultSegmentorV2's seg_head
        if semantic_out_channels != boundary_feature_channels:
            self.semantic_remap_head = nn.Linear(boundary_feature_channels, semantic_out_channels)
        else:
            self.semantic_remap_head = nn.Identity()


    def forward(self, point: Point):
        # f_o: PTv3 디코더의 최종 출력 feature (N, in_channels)
        fo = point.feat 

        # 1. Decouple initial semantic and boundary features
        sem_out_init = self.semantic_init_mlp(fo)
        margin_out_init = self.boundary_init_mlp(fo)

        # 2. Generate QKV for Attention
        qkv_s_raw = self.sem_qkv(sem_out_init)
        qkv_m_raw = self.margin_qkv(margin_out_init)

        # Split into Q, K, V (each N, boundary_feature_channels)
        q_s, k_s, v_s = torch.chunk(qkv_s_raw, 3, dim=-1)
        q_m, k_m, v_m = torch.chunk(qkv_m_raw, 3, dim=-1)

        # 3. Fuse Queries (following BFANet Eq 6: Ct(Qb, Qs) -> Mf1)
        fused_query = self.fusion_q(torch.cat((q_s, q_m), dim=-1))

        # 4. Reshape for Multi-Head Attention Calculation (K=1, single token per point)
        q_all_for_attn = fused_query.view(fo.shape[0], self.num_heads, 1, self.head_dim)
        k_s_for_attn = k_s.view(fo.shape[0], self.num_heads, 1, self.head_dim)
        v_s_for_attn = v_s.view(fo.shape[0], self.num_heads, 1, self.head_dim)
        k_m_for_attn = k_m.view(fo.shape[0], self.num_heads, 1, self.head_dim)
        v_m_for_attn = v_m.view(fo.shape[0], self.num_heads, 1, self.head_dim)
        
        # 5. Perform Attention for Semantic Features (Eq 6)
        attn_sem = q_all_for_attn @ k_s_for_attn.transpose(-2, -1) * self.scale
        attn_sem = self.softmax(attn_sem)
        attn_sem = self.attn_drop(attn_sem)
        
        sem_out_fused = (attn_sem @ v_s_for_attn).view(fo.shape[0], -1) 
        
        # 6. Perform Attention for Boundary Features
        attn_bou = q_all_for_attn @ k_m_for_attn.transpose(-2, -1) * self.scale
        attn_bou = self.softmax(attn_bou)
        attn_bou = self.attn_drop(attn_bou)
        
        margin_out_fused = (attn_bou @ v_m_for_attn).view(fo.shape[0], -1)

        # 7. Final Boundary Prediction Logits
        boundary_pred_logits = self.boundary_pred_head(margin_out_fused)

        # 8. Remap semantic feature to desired output channel
        fs = self.semantic_remap_head(sem_out_fused)

        # Update Point object: point.feat for semantic head, and add boundary_pred_logits
        point.feat = fs
        point.boundary_pred_logits = boundary_pred_logits 

        return point 