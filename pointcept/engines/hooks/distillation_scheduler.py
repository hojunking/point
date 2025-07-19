# In pointcept/engines/hooks/distillation_scheduler.py

from pointcept.engines.hooks import HookBase
from .builder import HOOKS
from pointcept.models.losses.misc import FeatureDistillationLoss # Loss 클래스가 있는 정확한 경로로 수정 필요

@HOOKS.register_module()
class DistillationSchedulerHook(HookBase):
    def __init__(self,
                 start_epoch, 
                 distill_loss_weight):
        self.start_epoch = start_epoch
        self.distill_loss_weight = distill_loss_weight
        self.activated = False

    def before_epoch(self): # [수정 1] 'trainer' 인자 제거
        if not self.activated and self.trainer.epoch >= self.start_epoch:
            for criterion in self.trainer.model.module.criteria:
                if isinstance(criterion, FeatureDistillationLoss):
                    criterion.loss_weight = self.distill_loss_weight
                    self.trainer.logger.info(
                        f"Epoch {self.trainer.epoch}: Distillation loss weight ACTIVATED and set to {self.distill_loss_weight}."
                    )
                    self.activated = True # 활성화 상태로 변경
                    break 