"""
OctFormer + BFANet-style Boundary-Semantic Head

Built on top of OctFormer-v1m1 modules and adapted for BoundarySemanticLoss
(i.e., boundary outputs are logits, not sigmoid probabilities).
"""

from typing import Optional, List
import torch
import torch.nn as nn

try:
    import ocnn
    from ocnn.octree import Points
except ImportError:
    from pointcept.utils.misc import DummyClass

    ocnn = None
    Points = DummyClass

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
from .octformer_v1m1_base import (
    OctreeT,
    PatchEmbed,
    OctFormerStage,
    Downsample,
    OctFormerDecoder,
)


class OctFormerBFAHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        boundary_feature_channels: int = 128,
        num_heads: int = 8,
        dropout: List[float] = (0.0, 0.0),
    ):
        super().__init__()
        assert (
            boundary_feature_channels % num_heads == 0
        ), "boundary_feature_channels must be divisible by num_heads"

        self.boundary_feature_channels = boundary_feature_channels
        self.num_heads = num_heads
        self.head_dim = boundary_feature_channels // num_heads
        self.scale = self.head_dim**-0.5

        self.semantic_init_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, boundary_feature_channels),
            nn.BatchNorm1d(boundary_feature_channels),
            nn.LeakyReLU(),
        )
        self.boundary_init_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, boundary_feature_channels),
            nn.BatchNorm1d(boundary_feature_channels),
            nn.LeakyReLU(),
        )

        self.initial_semantic_head = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(boundary_feature_channels, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(64, num_classes, bias=True),
        )
        self.initial_boundary_head = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(boundary_feature_channels, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(64, 1, bias=True),
        )

        self.sem_qkv_proj = nn.Sequential(
            nn.Linear(boundary_feature_channels, boundary_feature_channels * 3),
            nn.BatchNorm1d(boundary_feature_channels * 3),
            nn.LeakyReLU(),
        )
        self.boundary_qkv_proj = nn.Sequential(
            nn.Linear(boundary_feature_channels, boundary_feature_channels * 3),
            nn.BatchNorm1d(boundary_feature_channels * 3),
            nn.LeakyReLU(),
        )

        self.fusion_q_proj = nn.Sequential(
            nn.Linear(boundary_feature_channels * 2, boundary_feature_channels * 2),
            nn.BatchNorm1d(boundary_feature_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(boundary_feature_channels * 2, boundary_feature_channels),
            nn.BatchNorm1d(boundary_feature_channels),
            nn.LeakyReLU(),
        )

        self.final_semantic_head = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(boundary_feature_channels, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(64, num_classes, bias=True),
        )
        self.final_boundary_head = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(boundary_feature_channels, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(64, 1, bias=True),
        )

        self.attn_drop = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat: torch.Tensor):
        sem_out_init = self.semantic_init_mlp(feat)
        margin_out_init = self.boundary_init_mlp(feat)

        initial_sem_logits = self.initial_semantic_head(sem_out_init)
        initial_bou_logits = self.initial_boundary_head(margin_out_init)

        k_tokens = 1

        qkv_s = self.sem_qkv_proj(sem_out_init).reshape(
            -1, k_tokens, 3, self.num_heads, self.head_dim
        )
        qkv_s = qkv_s.permute(2, 0, 3, 1, 4)
        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]

        qkv_m = self.boundary_qkv_proj(margin_out_init).reshape(
            -1, k_tokens, 3, self.num_heads, self.head_dim
        )
        qkv_m = qkv_m.permute(2, 0, 3, 1, 4)
        q_m, k_m, v_m = qkv_m[0], qkv_m[1], qkv_m[2]

        q_all = torch.cat((q_s, q_m), dim=-1).reshape(feat.shape[0], -1)
        q_all = self.fusion_q_proj(q_all)
        q_all = q_all.reshape(feat.shape[0], self.num_heads, k_tokens, self.head_dim)

        attn_sem = q_all @ k_s.transpose(-2, -1) * self.scale
        attn_sem = self.softmax(attn_sem)
        attn_sem = self.attn_drop(attn_sem)
        sem_out_fused = (attn_sem @ v_s).view(feat.shape[0], -1)

        attn_bou = q_all @ k_m.transpose(-2, -1) * self.scale
        attn_bou = self.softmax(attn_bou)
        attn_bou = self.attn_drop(attn_bou)
        margin_out_fused = (attn_bou @ v_m).view(feat.shape[0], -1)

        final_sem_logits = self.final_semantic_head(sem_out_fused)
        final_bou_logits = self.final_boundary_head(margin_out_fused)

        return (
            initial_sem_logits,
            initial_bou_logits,
            final_sem_logits,
            final_bou_logits,
        )


@MODELS.register_module("OctFormer-v1m2-BS")
class OctFormerBS(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        fpn_channels=168,
        channels=(96, 192, 384, 384),
        num_blocks=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 24),
        patch_size=26,
        stem_down=2,
        head_up=2,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        octree_scale_factor=10.24,
        octree_depth=11,
        octree_full_depth=2,
        bfa_head_cfg=None,
    ):
        super().__init__()
        assert ocnn is not None, "Please follow `README.md` to install ocnn.`"

        self.patch_size = patch_size
        self.dilation = dilation
        self.nempty = nempty
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        self.octree_scale_factor = octree_scale_factor
        self.octree_depth = octree_depth
        self.octree_full_depth = octree_full_depth
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        self.layers = nn.ModuleList(
            [
                OctFormerStage(
                    dim=channels[i],
                    num_heads=num_heads[i],
                    patch_size=patch_size,
                    drop_path=drop_ratio[
                        sum(num_blocks[:i]) : sum(num_blocks[: i + 1])
                    ],
                    dilation=dilation,
                    nempty=nempty,
                    num_blocks=num_blocks[i],
                )
                for i in range(self.num_stages)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                Downsample(channels[i], channels[i + 1], kernel_size=[2], nempty=nempty)
                for i in range(self.num_stages - 1)
            ]
        )
        self.decoder = OctFormerDecoder(
            channels=channels, fpn_channel=fpn_channels, nempty=nempty, head_up=head_up
        )
        self.interp = ocnn.nn.OctreeInterp("nearest", nempty)

        if bfa_head_cfg is None:
            bfa_head_cfg = {}
        self.bfa_head = OctFormerBFAHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            boundary_feature_channels=bfa_head_cfg.get("boundary_feature_channels", 128),
            num_heads=bfa_head_cfg.get("num_heads", 8),
            dropout=bfa_head_cfg.get("dropout", (0.0, 0.0)),
        )

    def forward(self, data_dict):
        coord = data_dict["coord"]
        normal = data_dict["normal"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)

        point = Points(
            points=coord / self.octree_scale_factor,
            normals=normal,
            features=feat,
            batch_id=batch.unsqueeze(-1),
            batch_size=len(offset),
        )
        octree = ocnn.octree.Octree(
            depth=self.octree_depth,
            full_depth=self.octree_full_depth,
            batch_size=len(offset),
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()

        feat = self.patch_embed(octree.features[octree.depth], octree, octree.depth)
        depth = octree.depth - self.stem_down
        octree = OctreeT(
            octree,
            self.patch_size,
            self.dilation,
            self.nempty,
            max_depth=depth,
            start_depth=depth - self.num_stages + 1,
        )

        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            feat = self.layers[i](feat, octree, depth_i)
            features[depth_i] = feat
            if i < self.num_stages - 1:
                feat = self.downsamples[i](feat, octree, depth_i)

        out = self.decoder(features, octree)
        query_pts = torch.cat([point.points, point.batch_id], dim=1).contiguous()
        out = self.interp(out, octree, octree.depth, query_pts)

        (
            initial_semantic_logits,
            initial_boundary_logits,
            final_semantic_logits,
            final_boundary_logits,
        ) = self.bfa_head(out)

        return dict(
            initial_semantic_logits=initial_semantic_logits,
            initial_boundary_logits=initial_boundary_logits,
            final_semantic_logits=final_semantic_logits,
            final_boundary_logits=final_boundary_logits,
        )
