"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            loss += c(pred, target)
        return loss
    
class Criteria_matterport3d(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occurs in model
            return pred

        total_loss = None
        for c in self.criteria:
            # 각 loss의 ignore_index 가져오기 (loss마다 다를 수 있음)
            ignore_index = getattr(c, "ignore_index", -1)

            # 모든 픽셀이 ignore면 skip
            if (target == ignore_index).all():
                continue

            loss_val = c(pred, target)

            if total_loss is None:
                total_loss = loss_val
            else:
                total_loss = total_loss + loss_val

        # 모든 loss가 skip된 경우 — 반드시 Tensor로 반환해야 함!
        if total_loss is None:
            total_loss = pred.new_tensor(0.0, requires_grad=True)

        return total_loss

class Criteria_bs(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, initial_sem_logits, 
                 initial_bou_logits, 
                 final_sem_logits, 
                 final_bou_logits, 
                 gt_semantic_label, gt_boundary_label):
        
        loss = self.criteria[0](initial_sem_logits, 
                                initial_bou_logits, 
                                final_sem_logits, 
                                final_bou_logits,
                                gt_semantic_label, gt_boundary_label)
        return loss

class Criteria_distil(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        # config에 정의된 모든 loss들을 빌드합니다.
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))
    
    def __call__(self, 
                 seg_logits,             # Student의 분할 예측 로짓
                 student_feature_bridged,  # MLP 브릿지를 통과한 Student 특징
                 teacher_feature,        # Teacher 인코더의 특징
                 gt_semantic_label):     # 분할 정답 레이블
        
        total_semantic_loss = 0
        # self.criteria 리스트에 있는 모든 손실 함수를 순회하며 계산
        total_semantic_loss += self.criteria[0](seg_logits, gt_semantic_label)
        total_semantic_loss += self.criteria[1](seg_logits, gt_semantic_label)
        
        if student_feature_bridged is not None and teacher_feature is not None:
            distill_loss = self.criteria[2](student_feature_bridged, teacher_feature)
        else:
            distill_loss = 0.0
            
        return total_semantic_loss, distill_loss # 최종 합산된 스칼라 손실 값만 반환
    
class Criteria_bs_distil(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        # config에 정의된 모든 loss들을 빌드합니다.
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))
    
    def __call__(self,
                 point_student,          # [수정] Point 객체를 직접 받음
                 student_feature_bridged,
                 teacher_feature,
                 input_dict):            # gt_label을 포함한 dict
        
        bs_loss_dict = self.criteria[0](
            initial_sem_logits=point_student.initial_semantic_logits, 
            initial_bou_logits=point_student.initial_boundary_logits, 
            final_sem_logits=point_student.final_semantic_logits, 
            final_bou_logits=point_student.final_boundary_logits,
            gt_semantic_label=input_dict["segment"], 
            gt_boundary_label=input_dict["boundary"]
        )
        
        if student_feature_bridged is not None and teacher_feature is not None:
            distill_loss = self.criteria[1](student_feature_bridged, teacher_feature)
        else:
            distill_loss = 0.0

        # 최종 손실 딕셔너리 구성
        final_loss = bs_loss_dict
        final_loss['loss_distill'] = distill_loss
        final_loss['loss'] = bs_loss_dict['loss'] + distill_loss
        return final_loss

class Criteria_matterport3d_bs_distil(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self,
                 point_student,          # Point 객체 (logits 내장)
                 student_feature_bridged,
                 teacher_feature,
                 input_dict):
        bs_loss_dict = {}
        if len(self.criteria) > 0:
            bs_criterion = self.criteria[0]

            # Point 내부 logits과 GT label을 이용
            pred_sem_init = point_student.initial_semantic_logits
            pred_bou_init = point_student.initial_boundary_logits
            pred_sem_final = point_student.final_semantic_logits
            pred_bou_final = point_student.final_boundary_logits

            gt_sem = input_dict["segment"]
            gt_bou = input_dict["boundary"]

            ignore_index_sem = getattr(bs_criterion, "ignore_index", -1)
            ignore_index_bou = getattr(bs_criterion, "ignore_index", -1)

            if not (gt_sem == ignore_index_sem).all():
                num_classes = pred_sem_final.shape[1]
                gt_sem = gt_sem.clone()
                # out-of-range or invalid → ignore index로 통일
                gt_sem[(gt_sem < 0) | (gt_sem >= num_classes)] = ignore_index_sem

                bs_loss_dict = bs_criterion(
                    initial_sem_logits=pred_sem_init,
                    initial_bou_logits=pred_bou_init,
                    final_sem_logits=pred_sem_final,
                    final_bou_logits=pred_bou_final,
                    gt_semantic_label=gt_sem,
                    gt_boundary_label=gt_bou,
                )
            else:
                # loss가 모두 skip된 경우 tensor로 반환
                zero_tensor = pred_sem_final.new_tensor(0.0, requires_grad=True)
                bs_loss_dict = {"loss": zero_tensor}
        distill_loss = 0.0
        if len(self.criteria) > 1:
            distill_criterion = self.criteria[1]
            if student_feature_bridged is not None and teacher_feature is not None:
                distill_loss = distill_criterion(student_feature_bridged, teacher_feature)
            else:
                # 평가 시 None이면 0.0 loss 반환
                distill_loss = pred_sem_final.new_tensor(0.0, requires_grad=True)

        final_loss = bs_loss_dict
        final_loss["loss_distill"] = distill_loss
        final_loss["loss"] = bs_loss_dict["loss"] + distill_loss

        return final_loss
  
def build_criteria(cfg):
    return Criteria(cfg)

def build_matterport3d_criteria(cfg):
    return Criteria_matterport3d(cfg)

def build_criteria_bs(cfg):
    return Criteria_bs(cfg)

def build_criteria_distil(cfg):
    return Criteria_distil(cfg)

def build_criteria_bs_distil(cfg):
    return Criteria_bs_distil(cfg)

def build_matterport3d_criteria_bs_distil(cfg):
    return Criteria_matterport3d_bs_distil(cfg)