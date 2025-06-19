"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss



def _binary_cross_entropy(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target.float())

# BoundarySemanticLoss 내부에서 사용할 Binary Dice Loss 함수
def _binary_dice_loss(pred, target, smooth=1e-5, exponent=2):
    pred_prob = torch.sigmoid(pred) # 로짓을 확률로 변환
    target = target.float()
    
    intersection = (pred_prob * target).sum()
    union = (pred_prob.pow(exponent).sum() + target.pow(exponent).sum())
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return (1 - dice)

@LOSSES.register_module()
class BoundarySemanticLoss(nn.Module):
    def __init__(self, semantic_loss_weight=1.0, boundary_loss_weight=1.0,
                 ignore_index=-1, num_semantic_classes=None):
        super().__init__()
        self.semantic_loss_weight = semantic_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.ignore_index = ignore_index
        self.num_semantic_classes = num_semantic_classes if num_semantic_classes is not None else 20 
        
        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        self.dice_loss_semantic = DiceLoss(ignore_index=ignore_index, exponent=2)
        
        # Binary Cross Entropy 및 Binary Dice는 헬퍼 함수로 정의했으므로 별도 초기화 필요 없음.
        # 만약 BinaryFocalLoss를 사용하고 싶다면, 이곳에 self.bce_loss = BinaryFocalLoss(...) 등으로 초기화 가능.

    def forward(self, seg_logits, gt_semantic_label, boundary_logits, gt_boundary_label):
        # 1. Semantic Loss (CrossEntropy + Dice)
        loss_sem_ce = self.ce_loss(seg_logits, gt_semantic_label)
        loss_sem_dice = self.dice_loss_semantic(seg_logits, gt_semantic_label)
        
        total_loss_semantic = loss_sem_ce + loss_sem_dice # BFANet 논문의 L_sem
        
        losses = {'loss_semantic': total_loss_semantic * self.semantic_loss_weight}

        if gt_boundary_label.dim() == 1:
            gt_boundary_label = gt_boundary_label.unsqueeze(-1) # (N,) -> (N,1)
        
        # --- Boundary Loss를 위한 ignore_index 처리 시작 ---
        # ignore_index에 해당하지 않는 유효한 포인트들을 마스킹합니다.
        valid_boundary_mask = (gt_boundary_label != self.ignore_index).squeeze(-1) # (N,1) -> (N,)
        
        # 유효한 포인트에 해당하는 예측 로짓과 GT 레이블만 추출합니다.
        valid_boundary_logits = boundary_logits[valid_boundary_mask]
        valid_gt_boundary_label = gt_boundary_label[valid_boundary_mask]

        # 유효한 boundary 포인트가 하나도 없는 극단적인 배치인 경우
        if valid_gt_boundary_label.numel() == 0:
            # 해당 배치에 대한 Boundary Loss는 0으로 처리하여 NaN을 방지합니다.
            loss_bou_bce = torch.tensor(0.0, device=boundary_logits.device)
            loss_bou_dice = torch.tensor(0.0, device=boundary_logits.device)
        else:
            # 유효한 포인트에 대해서만 Loss를 계산합니다.
            loss_bou_bce = _binary_cross_entropy(valid_boundary_logits, valid_gt_boundary_label)
            loss_bou_dice = _binary_dice_loss(valid_boundary_logits, valid_gt_boundary_label)
        # --- ignore_index 처리 끝 ---

        total_loss_boundary = loss_bou_bce + loss_bou_dice 
        
        losses['loss_boundary'] = total_loss_boundary * self.boundary_loss_weight
        losses['loss'] = sum(_loss for _loss in losses.values())
        
        return losses