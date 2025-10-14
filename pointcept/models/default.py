import torch
import torch.nn as nn
import torch_scatter
import torch_cluster

from pointcept.models.losses import build_criteria, build_criteria_bs, build_criteria_distil, build_criteria_bs_distil, build_matterport3d_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model
from pointcept.models.utils import load_checkpoint

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
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
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict

@MODELS.register_module()
class matterport3dSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_matterport3d_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
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
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            #loss = self.criteria(seg_logits, input_dict["segment"])
            #return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict

@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)

@MODELS.register_module()
class SegmentorBS(nn.Module): # DefaultSegmentorV2를 대체할 새로운 클래스
    def __init__(
        self,
        num_classes,
        backbone_out_channels, # BFABlock의 semantic_out_channels와 일치해야 함.
        backbone=None,
        criteria=None, # 이제 BoundarySemanticLoss의 Config를 받게 됨.
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = nn.Identity() 
        self.backbone = build_model(backbone) # PT-v3m3-BSBlock 모델이 여기에 빌드됨.
        self.criteria = build_criteria_bs(criteria) # BoundarySemanticLoss 인스턴스가 빌드됨.
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point) # PT-v3m3-BSBlock의 forward 호출

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat # 최종 semantic feature (seg_head의 입력)
        else:
            feat = point # should not happen if backbone returns Point
        
        #seg_logits = self.seg_head(feat) # 최종 semantic prediction logits (N, num_classes)
        
        return_dict = dict()
        if return_point:
            return_dict["point"] = point # point 객체 (boundary_pred_logits 포함) 반환

        # 모델이 학습 모드일 때 (Loss 계산 필수)
        if self.training:
            losses_dict = self.criteria(
                initial_sem_logits=point.initial_semantic_logits,     
                initial_bou_logits=point.initial_boundary_logits,     
                final_sem_logits=point.final_semantic_logits,
                final_bou_logits=point.final_boundary_logits,
                gt_semantic_label=input_dict["segment"],
                gt_boundary_label=input_dict["boundary"]
            )
            return_dict.update(losses_dict)

        # 모델이 평가 또는 테스트 모드일 때
        else: # not self.training
            # 평가 모드 (GT semantic label이 input_dict에 있는 경우)
            if "segment" in input_dict.keys():
                # Loss 계산 (훈련과 동일한 방식, 단 gradient 계산은 없음)
                losses_dict = self.criteria(
                    initial_sem_logits=point.initial_semantic_logits,     
                    initial_bou_logits=point.initial_boundary_logits,     
                    final_sem_logits=point.final_semantic_logits,
                    final_bou_logits=point.final_boundary_logits,
                    gt_semantic_label=input_dict["segment"],
                    gt_boundary_label=input_dict["boundary"]
                )
                return_dict.update(losses_dict)

            # 예측 결과는 항상 반환 (평가 및 테스트 모드)
            return_dict["seg_logits"] = point.final_semantic_logits 
            return_dict["boundary_logits"] = point.final_boundary_logits 

        return return_dict
    
@MODELS.register_module()
class SegmentorDistill(nn.Module):
    def __init__(self,
                 backbone,              # Student 백본 config (PT-v3)
                 teacher_backbone,      # Teacher 백본 config (사전 학습된 PT-v3)
                 mlp_bridge,            # MLP 브릿지 config
                 criteria,              # 분할 Loss + 증류 Loss config 리스트
                 num_classes,
                 backbone_out_channels,
                 freeze_backbone=False):
        super().__init__()
        
        # 1. Student 모델 초기화 (DefaultSegmentorV2와 동일)
        self.backbone = build_model(backbone)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        
        
        # 2. Teacher 모델 초기화
        teacher_backbone_cfg = teacher_backbone.copy()
        checkpoint_path = teacher_backbone_cfg.pop("checkpoint_path", None)
        
        self.teacher_backbone = build_model(teacher_backbone_cfg)
        
        if checkpoint_path:
            load_checkpoint(self.teacher_backbone, checkpoint_path, map_location="cpu")
        self.teacher_backbone.eval()
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
            
        # 3. MLP 브릿지 초기화
        self.mlp_bridge = nn.Sequential(
            nn.Linear(mlp_bridge["in_channels"], mlp_bridge["hidden_channels"]),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_bridge["hidden_channels"], mlp_bridge["out_channels"])
        )

        self.criteria = build_criteria_distil(criteria)

    def forward(self, input_dict):
        # 1. Student 경로 실행
        point_student = Point(input_dict)
        point_student, student_feature_for_distill = self.backbone(point_student)
        
        if isinstance(point_student, Point):
            while "pooling_parent" in point_student.keys():
                assert "pooling_inverse" in point_student.keys()
                parent = point_student.pop("pooling_parent")
                inverse = point_student.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point_student.feat[inverse]], dim=-1)
                point_student = parent
            feat_for_seg = point_student.feat
        else:
            feat_for_seg = point_student # should not happen

        seg_logits = self.seg_head(feat_for_seg)
        
        # 2. Teacher 경로 실행 (그래디언트 계산 없이)
        with torch.no_grad():
            teacher_input_dict = {
                "coord": input_dict["coord"],
                "grid_coord": input_dict["grid_coord"],
                "offset": input_dict["offset"],
                "feat": input_dict["feat_teacher"],
                "scene_name": input_dict.get("scene_name")
            }
            point_teacher = Point(teacher_input_dict)
            point_teacher.serialization(order=self.teacher_backbone.order, shuffle_orders=self.teacher_backbone.shuffle_orders)
            point_teacher.sparsify()
            point_teacher = self.teacher_backbone.embedding(point_teacher)
            point_teacher = self.teacher_backbone.enc(point_teacher) # 인코더까지만 실행
            teacher_feature_for_distill = point_teacher.feat
        
        # 3. 특징 증류
        student_feature_bridged = self.mlp_bridge(student_feature_for_distill)
        
        # 4. 손실 계산 및 반환
        return_dict = {}
        if self.training:
            total_semantic_loss, distil_loss = self.criteria(
                seg_logits=seg_logits,
                student_feature_bridged=student_feature_bridged,
                teacher_feature=teacher_feature_for_distill,
                gt_semantic_label=input_dict["segment"]
            )
            
            return_dict["loss_seg"] = total_semantic_loss
            return_dict["loss_distill"] = distil_loss
            return_dict["loss"] = total_semantic_loss + distil_loss
        
        else: # 평가/테스트 모드
            if "segment" in input_dict:
                total_semantic_loss, _ = self.criteria(
                    seg_logits=seg_logits,
                    student_feature_bridged=None,
                    teacher_feature=None,
                    gt_semantic_label=input_dict["segment"]
                )
                
                return_dict["loss"] = total_semantic_loss
            return_dict["seg_logits"] = seg_logits
            
        return return_dict


@MODELS.register_module()
class SegmentorBSDistill(nn.Module):
    def __init__(self,
                 backbone,              # Student 백본 config (PT-v3 + BSBlock)
                 teacher_backbone,      # Teacher 백본 config (사전 학습된 PT-v3)
                 mlp_bridge,            # MLP 브릿지 config
                 criteria,              # 모든 Loss config를 담은 리스트
                 **kwargs):
        super().__init__()
        
        # 1. Student 모델 초기화 (PT-v3 + BSBlock)
        self.backbone = build_model(backbone)
        
        # 2. Teacher 모델 초기화 (사전 학습된 인코더)
        teacher_backbone_cfg = teacher_backbone.copy()
        checkpoint_path = teacher_backbone_cfg.pop("checkpoint_path", None)
        self.teacher_backbone = build_model(teacher_backbone_cfg)
        if checkpoint_path:
            load_checkpoint(self.teacher_backbone, checkpoint_path, map_location="cpu")
        self.teacher_backbone.eval()
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
            
        # 3. MLP 브릿지 초기화
        self.mlp_bridge = nn.Sequential(
            nn.Linear(mlp_bridge["in_channels"], mlp_bridge["hidden_channels"]),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_bridge["hidden_channels"], mlp_bridge["out_channels"])
        )

        # 4. 모든 손실 함수를 관리할 Criteria 객체 생성
        self.criteria = build_criteria_bs_distil(criteria)

    def forward(self, input_dict, return_point=False):
        # 1. Student 경로 실행
        # backbone은 PTv3-BSBlock이며, 최종적으로 4종류의 logit과 인코더 특징을 담은 Point 객체를 반환해야 함
        point_student, student_feature_for_distill = self.backbone(Point(input_dict))
        
        if isinstance(point_student, Point):
            while "pooling_parent" in point_student.keys():
                assert "pooling_inverse" in point_student.keys()
                parent = point_student.pop("pooling_parent")
                inverse = point_student.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point_student.feat[inverse]], dim=-1)
                point_student = parent
            feat_for_seg = point_student.feat
        else:
            feat_for_seg = point_student # should not happen
        
        
        # 2. Teacher 경로 실행 (그래디언트 계산 없이)
        with torch.no_grad():
            teacher_input_dict = {
                "coord": input_dict["coord"],
                "grid_coord": input_dict["grid_coord"],
                "offset": input_dict["offset"],
                "feat": input_dict["feat_teacher"],
                "scene_name": input_dict.get("scene_name")
            }
            point_teacher = Point(teacher_input_dict)
            point_teacher.serialization(order=self.teacher_backbone.order, shuffle_orders=self.teacher_backbone.shuffle_orders)
            point_teacher.sparsify()
            point_teacher = self.teacher_backbone.embedding(point_teacher)
            point_teacher = self.teacher_backbone.enc(point_teacher) # 인코더까지만 실행
            teacher_feature_for_distill = point_teacher.feat

        # 3. 특징 증류
        student_feature_bridged = self.mlp_bridge(student_feature_for_distill)
        
        return_dict = dict()
        if return_point:
            return_dict["point"] = point_student # point 객체 (boundary_pred_logits 포함) 반환
        
        if self.training:
            losses_dict = self.criteria(
                point_student=point_student,
                student_feature_bridged=student_feature_bridged,
                teacher_feature=teacher_feature_for_distill,
                input_dict=input_dict
            )
            return_dict.update(losses_dict)
        
        else: # 평가 모드
            # 평가 시에는 메인 손실(경계+분할)만 계산
            if "segment" in input_dict.keys():
                losses_dict = self.criteria(
                    point_student=point_student,
                    student_feature_bridged=None,
                    teacher_feature=None,
                    input_dict=input_dict
                )
                return_dict = losses_dict
            
            return_dict["seg_logits"] = point_student.final_semantic_logits
            return_dict["boundary_logits"] = point_student.final_boundary_logits
        return return_dict