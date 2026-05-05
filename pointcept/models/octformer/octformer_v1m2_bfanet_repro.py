"""
OctFormer + BFANet reproduction head.

This module keeps the existing Pointcept boundary dataset contract, but mirrors
BFANet's OctFormer segmentation head: the head consumes the multi-scale
OctFormer feature dict, performs FPN decoding/interpolation, then predicts
initial/final semantic scores and sigmoid boundary probabilities.
"""

from typing import Dict, List

import torch
import torch.nn as nn

try:
    import ocnn
    import dwconv
    from ocnn.octree import Points
except ImportError:
    from pointcept.utils.misc import DummyClass

    ocnn = None
    dwconv = None
    Points = DummyClass

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
from .octformer_v1m1_base import OctreeT, PatchEmbed, OctFormerStage, Downsample


class BFANetReproHead(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: List[int],
        fpn_channel: int,
        nempty: bool,
        num_up: int = 1,
        dropout: List[float] = (0.0, 0.0),
    ):
        super().__init__()
        self.num_up = num_up
        self.num_stages = len(channels)

        self.conv1x1 = nn.ModuleList(
            [
                nn.Linear(channels[i], fpn_channel)
                for i in range(self.num_stages - 1, -1, -1)
            ]
        )
        self.upsample = ocnn.nn.OctreeUpsample("nearest", nempty)
        self.conv3x3 = nn.ModuleList(
            [
                ocnn.modules.OctreeConvBnRelu(
                    fpn_channel, fpn_channel, kernel_size=[3], stride=1, nempty=nempty
                )
                for _ in range(self.num_stages)
            ]
        )
        self.up_conv = nn.ModuleList(
            [
                ocnn.modules.OctreeDeconvBnRelu(
                    fpn_channel, fpn_channel, kernel_size=[3], stride=2, nempty=nempty
                )
                for _ in range(self.num_up)
            ]
        )
        self.interp = ocnn.nn.OctreeInterp("nearest", nempty)

        self.classifier_feat = nn.Sequential(
            nn.Linear(fpn_channel, fpn_channel),
            nn.BatchNorm1d(fpn_channel),
            nn.LeakyReLU(),
            nn.Linear(fpn_channel, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.margin_feat = nn.Sequential(
            nn.Linear(fpn_channel, fpn_channel),
            nn.BatchNorm1d(fpn_channel),
            nn.LeakyReLU(),
            nn.Linear(fpn_channel, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(64, out_channels, bias=True),
        )
        self.margin_score = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(64, 1, bias=True),
            nn.Sigmoid(),
        )
        self.classifier_v2 = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(64, out_channels, bias=True),
        )
        self.margin_score_v2 = nn.Sequential(
            nn.Dropout(dropout[0]),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout[1]),
            nn.Linear(64, 1, bias=True),
            nn.Sigmoid(),
        )

        self.scale = 1.0
        self.sem_qkv = nn.Sequential(
            nn.Linear(128, 128 * 3, bias=True),
            nn.BatchNorm1d(128 * 3),
            nn.LeakyReLU(),
        )
        self.margin_qkv = nn.Sequential(
            nn.Linear(128, 128 * 3, bias=True),
            nn.BatchNorm1d(128 * 3),
            nn.LeakyReLU(),
        )
        self.attn_drop = nn.Dropout(0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.fusion_q = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )

    def forward(
        self, features: Dict[int, torch.Tensor], octree, query_pts: torch.Tensor
    ):
        depth = min(features.keys())
        depth_max = max(features.keys())
        assert self.num_stages == len(features)

        feature = self.conv1x1[0](features[depth])
        conv_out = self.conv3x3[0](feature, octree, depth)
        out = self.upsample(conv_out, octree, depth, depth_max)
        for i in range(1, self.num_stages):
            depth_i = depth + i
            feature = self.upsample(feature, octree, depth_i - 1)
            feature = self.conv1x1[i](features[depth_i]) + feature
            conv_out = self.conv3x3[i](feature, octree, depth_i)
            out = out + self.upsample(conv_out, octree, depth_i, depth_max)
        for i in range(self.num_up):
            out = self.up_conv[i](out, octree, depth_max + i)

        out = self.interp(out, octree, depth_max + self.num_up, query_pts)

        sem_out = self.classifier_feat(out)
        margin_out = self.margin_feat(out)

        sem_score = self.classifier(sem_out)
        margin_score = self.margin_score(margin_out)

        k_tokens, num_heads, channels = 1, 8, 128
        qkv_s = self.sem_qkv(sem_out).reshape(
            -1, k_tokens, 3, num_heads, channels // num_heads
        )
        qkv_s = qkv_s.permute(2, 0, 3, 1, 4)
        qkv_m = self.margin_qkv(margin_out).reshape(
            -1, k_tokens, 3, num_heads, channels // num_heads
        )
        qkv_m = qkv_m.permute(2, 0, 3, 1, 4)

        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]
        q_m, k_m, v_m = qkv_m[0], qkv_m[1], qkv_m[2]
        cat_q = torch.cat((q_s, q_m), dim=-1).reshape(-1, channels * 2)
        q_all = self.fusion_q(cat_q).reshape(
            -1, num_heads, k_tokens, channels // num_heads
        )

        attn = q_all @ k_s.transpose(-2, -1) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        sem_out = (attn @ v_s).transpose(1, 2).reshape(-1, channels)

        attn = q_all @ k_m.transpose(-2, -1) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        margin_out = (attn @ v_m).transpose(1, 2).reshape(-1, channels)

        sem_score_v2 = self.classifier_v2(sem_out)
        margin_score_v2 = self.margin_score_v2(margin_out)

        return sem_score, margin_score, sem_score_v2, margin_score_v2


@MODELS.register_module("OctFormer-v1m2-BFANet")
class OctFormerBFANet(nn.Module):
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
        dropout=(0.0, 0.0),
    ):
        super().__init__()
        assert ocnn is not None, "Please follow `README.md` to install ocnn.`"
        assert dwconv is not None, "Please follow `README.md` to install dwconv.`"

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
        self.seg_header = BFANetReproHead(
            out_channels=num_classes,
            channels=channels,
            fpn_channel=fpn_channels,
            nempty=nempty,
            num_up=head_up,
            dropout=dropout,
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

        query_pts = torch.cat([point.points, point.batch_id], dim=1).contiguous()
        (
            initial_semantic_logits,
            initial_boundary_probs,
            final_semantic_logits,
            final_boundary_probs,
        ) = self.seg_header(features, octree, query_pts)

        return dict(
            initial_semantic_logits=initial_semantic_logits,
            initial_boundary_logits=initial_boundary_probs,
            final_semantic_logits=final_semantic_logits,
            final_boundary_logits=final_boundary_probs,
        )
