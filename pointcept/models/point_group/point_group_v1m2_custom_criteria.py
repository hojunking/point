"""
PointGroup for instance segmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Chengyao Wang
Please cite our work if the code is helpful to you.
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pointgroup_ops import ballquery_batch_p, bfs_cluster
except ImportError:
    ballquery_batch_p, bfs_cluster = None, None

from pointcept.models.utils import offset2batch, batch2offset
from pointcept.models.utils.structure import Point
from pointcept.models.utils import load_checkpoint

from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import build_criteria, build_criteria_bs


@MODELS.register_module("PG-v1m2")
class PointGroup(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        cluster_thresh=1.5,
        cluster_closed_points=300,
        cluster_propose_points=100,
        cluster_min_points=50,
        voxel_size=0.02,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.backbone = build_model(backbone)
        self.bias_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 3),
        )
        self.seg_head = nn.Linear(backbone_out_channels, semantic_num_classes)
        self.seg_criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, data_dict, return_point=False):
        if return_point:
            return dict(point=self.backbone(data_dict))
        coord = data_dict["coord"]
        instance_centroid = data_dict["instance_centroid"]
        offset = data_dict["offset"]

        point = self.backbone(data_dict)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        bias_pred = self.bias_head(feat)
        logit_pred = self.seg_head(feat)

        # compute loss
        if "segment" in data_dict.keys() and "instance" in data_dict.keys():
            segment = data_dict["segment"]
            instance = data_dict["instance"]
            seg_loss = self.seg_criteria(logit_pred, segment)

            mask = (instance != self.instance_ignore_index).float()
            bias_gt = instance_centroid - coord
            bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
            bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

            bias_pred_norm = bias_pred / (
                torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
            )
            bias_gt_norm = bias_gt / (
                torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8
            )
            cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
            bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
                torch.sum(mask) + 1e-8
            )

            loss = seg_loss + bias_l1_loss + bias_cosine_loss
            return_dict = dict(
                loss=loss,
                seg_loss=seg_loss,
                bias_l1_loss=bias_l1_loss,
                bias_cosine_loss=bias_cosine_loss,
            )
        else:
            # skip for test split
            return_dict = dict()

        if not self.training:
            center_pred = coord + bias_pred
            center_pred /= self.voxel_size
            logit_pred = F.softmax(logit_pred, dim=-1)
            segment_pred = torch.max(logit_pred, 1)[1]  # [n]
            # cluster
            mask = (
                ~torch.concat(
                    [
                        (segment_pred == index).unsqueeze(-1)
                        for index in self.segment_ignore_index
                    ],
                    dim=1,
                )
                .sum(-1)
                .bool()
            )

            if mask.sum() == 0:
                proposals_idx = torch.zeros(0).int()
                proposals_offset = torch.zeros(1).int()
            else:
                center_pred_ = center_pred[mask]
                segment_pred_ = segment_pred[mask]

                batch_ = offset2batch(offset)[mask]
                offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
                idx, start_len = ballquery_batch_p(
                    center_pred_,
                    batch_.int(),
                    offset_.int(),
                    self.cluster_thresh,
                    self.cluster_closed_points,
                )
                proposals_idx, proposals_offset = bfs_cluster(
                    segment_pred_.int().cpu(),
                    idx.cpu(),
                    start_len.cpu(),
                    self.cluster_min_points,
                )
                proposals_idx[:, 1] = (
                    mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
                )

            # get proposal
            proposals_pred = torch.zeros(
                (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
            )
            proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
            instance_pred = segment_pred[
                proposals_idx[:, 1][proposals_offset[:-1].long()].long()
            ]
            proposals_point_num = proposals_pred.sum(1)
            proposals_mask = proposals_point_num > self.cluster_propose_points
            proposals_pred = proposals_pred[proposals_mask]
            instance_pred = instance_pred[proposals_mask]

            pred_scores = []
            pred_classes = []
            pred_masks = proposals_pred.detach().cpu()
            for proposal_id in range(len(proposals_pred)):
                segment_ = proposals_pred[proposal_id]
                confidence_ = logit_pred[
                    segment_.bool(), instance_pred[proposal_id]
                ].mean()
                object_ = instance_pred[proposal_id]
                pred_scores.append(confidence_)
                pred_classes.append(object_)
            if len(pred_scores) > 0:
                pred_scores = torch.stack(pred_scores).cpu()
                pred_classes = torch.stack(pred_classes).cpu()
            else:
                pred_scores = torch.tensor([])
                pred_classes = torch.tensor([])

            return_dict["pred_scores"] = pred_scores
            return_dict["pred_masks"] = pred_masks
            return_dict["pred_classes"] = pred_classes
        return return_dict


@MODELS.register_module("PG-v1m3")
class PointGroupV1M3(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        cluster_thresh=1.5,
        cluster_closed_points=300,
        cluster_propose_points=100,
        cluster_min_points=50,
        voxel_size=0.02,
        criteria=None,
        criteria_bs=None,
        freeze_backbone=False,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.backbone = build_model(backbone)
        self.bias_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 3),
        )
        # keep for compatibility/fallback config parity with PG-v1m2
        self.seg_criteria = build_criteria(criteria)
        self.bs_criteria = build_criteria_bs(criteria_bs)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, data_dict, return_point=False):
        if return_point:
            return dict(point=self.backbone(data_dict))

        coord = data_dict["coord"]
        instance_centroid = data_dict["instance_centroid"]
        offset = data_dict["offset"]

        point = self.backbone(data_dict)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            raise TypeError(
                "PG-v1m3 expects backbone output as Point. "
                "Use PT-v3m3 bsblock backbone that returns Point with BS logits."
            )

        required_logits = (
            "initial_semantic_logits",
            "initial_boundary_logits",
            "final_semantic_logits",
            "final_boundary_logits",
        )
        missing_logits = [name for name in required_logits if not hasattr(point, name)]
        if len(missing_logits) > 0:
            raise RuntimeError(
                "PG-v1m3 requires PT-v3m3 bsblock backbone outputs with BS logits. "
                f"Missing attributes: {missing_logits}"
            )
        if "boundary" not in data_dict:
            raise KeyError(
                "PG-v1m3 requires input_dict['boundary'] for boundary supervision."
            )

        bias_pred = self.bias_head(feat)
        logit_pred = point.final_semantic_logits

        if "segment" in data_dict.keys() and "instance" in data_dict.keys():
            segment = data_dict["segment"]
            instance = data_dict["instance"]
            bs_loss_dict = self.bs_criteria(
                initial_sem_logits=point.initial_semantic_logits,
                initial_bou_logits=point.initial_boundary_logits,
                final_sem_logits=point.final_semantic_logits,
                final_bou_logits=point.final_boundary_logits,
                gt_semantic_label=segment,
                gt_boundary_label=data_dict["boundary"],
            )

            mask = (instance != self.instance_ignore_index).float()
            bias_gt = instance_centroid - coord
            bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
            bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

            bias_pred_norm = bias_pred / (
                torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
            )
            bias_gt_norm = bias_gt / (
                torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8
            )
            cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
            bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
                torch.sum(mask) + 1e-8
            )

            loss = bs_loss_dict["loss"] + bias_l1_loss + bias_cosine_loss
            return_dict = dict(
                loss=loss,
                bias_l1_loss=bias_l1_loss,
                bias_cosine_loss=bias_cosine_loss,
                loss_initial_semantic=bs_loss_dict.get("loss_initial_semantic"),
                loss_initial_boundary=bs_loss_dict.get("loss_initial_boundary"),
                loss_final_semantic=bs_loss_dict.get("loss_final_semantic"),
                loss_final_boundary=bs_loss_dict.get("loss_final_boundary"),
            )
        else:
            # skip for test split
            return_dict = dict()

        if not self.training:
            center_pred = coord + bias_pred
            center_pred /= self.voxel_size
            logit_pred = F.softmax(logit_pred, dim=-1)
            segment_pred = torch.max(logit_pred, 1)[1]  # [n]
            # cluster
            mask = (
                ~torch.concat(
                    [
                        (segment_pred == index).unsqueeze(-1)
                        for index in self.segment_ignore_index
                    ],
                    dim=1,
                )
                .sum(-1)
                .bool()
            )

            if mask.sum() == 0:
                proposals_idx = torch.zeros(0).int()
                proposals_offset = torch.zeros(1).int()
            else:
                center_pred_ = center_pred[mask]
                segment_pred_ = segment_pred[mask]

                batch_ = offset2batch(offset)[mask]
                offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
                idx, start_len = ballquery_batch_p(
                    center_pred_,
                    batch_.int(),
                    offset_.int(),
                    self.cluster_thresh,
                    self.cluster_closed_points,
                )
                proposals_idx, proposals_offset = bfs_cluster(
                    segment_pred_.int().cpu(),
                    idx.cpu(),
                    start_len.cpu(),
                    self.cluster_min_points,
                )
                proposals_idx[:, 1] = (
                    mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
                )

            # get proposal
            proposals_pred = torch.zeros(
                (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
            )
            proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
            instance_pred = segment_pred[
                proposals_idx[:, 1][proposals_offset[:-1].long()].long()
            ]
            proposals_point_num = proposals_pred.sum(1)
            proposals_mask = proposals_point_num > self.cluster_propose_points
            proposals_pred = proposals_pred[proposals_mask]
            instance_pred = instance_pred[proposals_mask]

            pred_scores = []
            pred_classes = []
            pred_masks = proposals_pred.detach().cpu()
            for proposal_id in range(len(proposals_pred)):
                segment_ = proposals_pred[proposal_id]
                confidence_ = logit_pred[
                    segment_.bool(), instance_pred[proposal_id]
                ].mean()
                object_ = instance_pred[proposal_id]
                pred_scores.append(confidence_)
                pred_classes.append(object_)
            if len(pred_scores) > 0:
                pred_scores = torch.stack(pred_scores).cpu()
                pred_classes = torch.stack(pred_classes).cpu()
            else:
                pred_scores = torch.tensor([])
                pred_classes = torch.tensor([])

            return_dict["seg_logits"] = point.final_semantic_logits
            return_dict["boundary_logits"] = point.final_boundary_logits
            return_dict["pred_scores"] = pred_scores
            return_dict["pred_masks"] = pred_masks
            return_dict["pred_classes"] = pred_classes
        return return_dict


@MODELS.register_module("PG-v1m4")
class PointGroupV1M4(nn.Module):
    def __init__(
        self,
        backbone,
        teacher_backbone,
        mlp_bridge,
        backbone_out_channels=64,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        cluster_thresh=1.5,
        cluster_closed_points=300,
        cluster_propose_points=100,
        cluster_min_points=50,
        voxel_size=0.02,
        criteria=None,
        criteria_bs=None,
        criteria_distill=None,
        freeze_backbone=False,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.backbone = build_model(backbone)
        self.bias_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 3),
        )
        # keep for compatibility/config parity with PG-v1m2/v1m3
        self.seg_criteria = build_criteria(criteria)
        self.bs_criteria = build_criteria_bs(criteria_bs)
        self.distill_criteria = build_criteria(criteria_distill)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        teacher_backbone_cfg = teacher_backbone.copy()
        checkpoint_path = teacher_backbone_cfg.pop("checkpoint_path", None)
        self.teacher_backbone = build_model(teacher_backbone_cfg)
        if checkpoint_path:
            load_checkpoint(self.teacher_backbone, checkpoint_path, map_location="cpu")
        self.teacher_backbone.eval()
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        self.mlp_bridge = nn.Sequential(
            nn.Linear(mlp_bridge["in_channels"], mlp_bridge["hidden_channels"]),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_bridge["hidden_channels"], mlp_bridge["out_channels"]),
        )

    def forward(self, data_dict, return_point=False):
        coord = data_dict["coord"]
        instance_centroid = data_dict["instance_centroid"]
        offset = data_dict["offset"]

        student_output = self.backbone(data_dict)
        if isinstance(student_output, tuple):
            point, student_feature_for_distill = student_output
        else:
            raise TypeError(
                "PG-v1m4 expects backbone output as (Point, encoder_feature). "
                "Use PT-v3m4 bs-distill backbone."
            )
        if return_point:
            return dict(point=point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            raise TypeError("PG-v1m4 expects first backbone output to be Point.")

        required_logits = (
            "initial_semantic_logits",
            "initial_boundary_logits",
            "final_semantic_logits",
            "final_boundary_logits",
        )
        missing_logits = [name for name in required_logits if not hasattr(point, name)]
        if len(missing_logits) > 0:
            raise RuntimeError(
                "PG-v1m4 requires BS logits on Point output. "
                f"Missing attributes: {missing_logits}"
            )
        if "boundary" not in data_dict:
            raise KeyError(
                "PG-v1m4 requires input_dict['boundary'] for boundary supervision."
            )

        bias_pred = self.bias_head(feat)
        logit_pred = point.final_semantic_logits

        if "segment" in data_dict.keys() and "instance" in data_dict.keys():
            segment = data_dict["segment"]
            instance = data_dict["instance"]
            bs_loss_dict = self.bs_criteria(
                initial_sem_logits=point.initial_semantic_logits,
                initial_bou_logits=point.initial_boundary_logits,
                final_sem_logits=point.final_semantic_logits,
                final_bou_logits=point.final_boundary_logits,
                gt_semantic_label=segment,
                gt_boundary_label=data_dict["boundary"],
            )

            mask = (instance != self.instance_ignore_index).float()
            bias_gt = instance_centroid - coord
            bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
            bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

            bias_pred_norm = bias_pred / (
                torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
            )
            bias_gt_norm = bias_gt / (
                torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8
            )
            cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
            bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
                torch.sum(mask) + 1e-8
            )

            distill_loss = point.feat.new_tensor(0.0)
            if self.training:
                if "feat_teacher" not in data_dict:
                    raise KeyError(
                        "PG-v1m4 training requires input_dict['feat_teacher'] for distillation."
                    )
                with torch.no_grad():
                    teacher_input_dict = {
                        "coord": data_dict["coord"],
                        "grid_coord": data_dict["grid_coord"],
                        "offset": data_dict["offset"],
                        "feat": data_dict["feat_teacher"],
                        "scene_name": data_dict.get("scene_name"),
                    }
                    point_teacher = Point(teacher_input_dict)
                    point_teacher.serialization(
                        order=self.teacher_backbone.order,
                        shuffle_orders=self.teacher_backbone.shuffle_orders,
                    )
                    point_teacher.sparsify()
                    point_teacher = self.teacher_backbone.embedding(point_teacher)
                    point_teacher = self.teacher_backbone.enc(point_teacher)
                    teacher_feature_for_distill = point_teacher.feat

                student_feature_bridged = self.mlp_bridge(student_feature_for_distill)
                if len(self.distill_criteria.criteria) > 0:
                    distill_loss = self.distill_criteria(
                        student_feature_bridged, teacher_feature_for_distill
                    )

            loss = bs_loss_dict["loss"] + bias_l1_loss + bias_cosine_loss + distill_loss
            return_dict = dict(
                loss=loss,
                bias_l1_loss=bias_l1_loss,
                bias_cosine_loss=bias_cosine_loss,
                loss_distill=distill_loss,
                loss_initial_semantic=bs_loss_dict.get("loss_initial_semantic"),
                loss_initial_boundary=bs_loss_dict.get("loss_initial_boundary"),
                loss_final_semantic=bs_loss_dict.get("loss_final_semantic"),
                loss_final_boundary=bs_loss_dict.get("loss_final_boundary"),
            )
        else:
            # skip for test split
            return_dict = dict()

        if not self.training:
            center_pred = coord + bias_pred
            center_pred /= self.voxel_size
            logit_pred = F.softmax(logit_pred, dim=-1)
            segment_pred = torch.max(logit_pred, 1)[1]  # [n]
            # cluster
            mask = (
                ~torch.concat(
                    [
                        (segment_pred == index).unsqueeze(-1)
                        for index in self.segment_ignore_index
                    ],
                    dim=1,
                )
                .sum(-1)
                .bool()
            )

            if mask.sum() == 0:
                proposals_idx = torch.zeros(0).int()
                proposals_offset = torch.zeros(1).int()
            else:
                center_pred_ = center_pred[mask]
                segment_pred_ = segment_pred[mask]

                batch_ = offset2batch(offset)[mask]
                offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
                idx, start_len = ballquery_batch_p(
                    center_pred_,
                    batch_.int(),
                    offset_.int(),
                    self.cluster_thresh,
                    self.cluster_closed_points,
                )
                proposals_idx, proposals_offset = bfs_cluster(
                    segment_pred_.int().cpu(),
                    idx.cpu(),
                    start_len.cpu(),
                    self.cluster_min_points,
                )
                proposals_idx[:, 1] = (
                    mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
                )

            # get proposal
            proposals_pred = torch.zeros(
                (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
            )
            proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
            instance_pred = segment_pred[
                proposals_idx[:, 1][proposals_offset[:-1].long()].long()
            ]
            proposals_point_num = proposals_pred.sum(1)
            proposals_mask = proposals_point_num > self.cluster_propose_points
            proposals_pred = proposals_pred[proposals_mask]
            instance_pred = instance_pred[proposals_mask]

            pred_scores = []
            pred_classes = []
            pred_masks = proposals_pred.detach().cpu()
            for proposal_id in range(len(proposals_pred)):
                segment_ = proposals_pred[proposal_id]
                confidence_ = logit_pred[
                    segment_.bool(), instance_pred[proposal_id]
                ].mean()
                object_ = instance_pred[proposal_id]
                pred_scores.append(confidence_)
                pred_classes.append(object_)
            if len(pred_scores) > 0:
                pred_scores = torch.stack(pred_scores).cpu()
                pred_classes = torch.stack(pred_classes).cpu()
            else:
                pred_scores = torch.tensor([])
                pred_classes = torch.tensor([])

            return_dict["seg_logits"] = point.final_semantic_logits
            return_dict["boundary_logits"] = point.final_boundary_logits
            return_dict["pred_scores"] = pred_scores
            return_dict["pred_masks"] = pred_masks
            return_dict["pred_classes"] = pred_classes
        return return_dict
