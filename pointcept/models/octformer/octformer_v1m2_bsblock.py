"""
OctFormer with the same BSBlock contract used by PT-v3m3-BSBlock.
"""

import torch
import torch.nn as nn

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3.point_transformer_v3m3_bsblock import BSBlock
from .octformer_v1m1_base import OctFormer, OctreeT


@MODELS.register_module("OctFormer-v1m2-BSBlock")
class OctFormerBSBlock(OctFormer):
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
        bsblock_cfg=None,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=0,
            fpn_channels=fpn_channels,
            channels=channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            patch_size=patch_size,
            stem_down=stem_down,
            head_up=head_up,
            dilation=dilation,
            drop_path=drop_path,
            nempty=nempty,
            octree_scale_factor=octree_scale_factor,
            octree_depth=octree_depth,
            octree_full_depth=octree_full_depth,
        )
        self.seg_head = nn.Identity()

        bsblock_cfg = dict(bsblock_cfg or {})
        bsblock_cfg.setdefault("in_channels", fpn_channels)
        bsblock_cfg.setdefault("semantic_out_channels", fpn_channels)
        bsblock_cfg.setdefault("num_semantic_classes", num_classes)
        self.bfanet_block = BSBlock(**bsblock_cfg)

    def forward(self, data_dict):
        point_dict = data_dict if isinstance(data_dict, Point) else Point(data_dict)
        coord = point_dict["coord"]
        normal = point_dict["normal"]
        feat = point_dict["feat"]
        offset = point_dict["offset"]
        batch = offset2batch(offset)

        point = self._points_cls(
            points=coord / self.octree_scale_factor,
            normals=normal,
            features=feat,
            batch_id=batch.unsqueeze(-1),
            batch_size=len(offset),
        )
        octree = self._octree_cls(
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

        point_dict.feat = out
        return self.bfanet_block(point_dict)

    @property
    def _points_cls(self):
        from ocnn.octree import Points

        return Points

    @property
    def _octree_cls(self):
        import ocnn

        return ocnn.octree.Octree
