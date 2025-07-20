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

    def before_epoch(self):
        # trainer.epoch는 0부터 시작합니다.
        if not self.activated and self.trainer.epoch >= self.start_epoch:
            # [핵심 수정] DDP 래핑 여부를 확인합니다.
            # hasattr(object, name)은 객체가 특정 속성을 가지고 있는지 확인하는 함수입니다.
            if hasattr(self.trainer.model, 'module'):
                # 다중 GPU 환경: .module을 통해 원래 모델에 접근
                model = self.trainer.model.module
            else:
                # 단일 GPU 환경: .module 없이 바로 접근
                model = self.trainer.model
            
            # 이제 model 변수를 사용하여 criteria에 접근합니다.
            for criterion in model.criteria:
                if isinstance(criterion, FeatureDistillationLoss):
                    criterion.loss_weight = self.distill_loss_weight
                    self.trainer.logger.info(
                        f"Epoch {self.trainer.epoch}: Distillation loss weight ACTIVATED and set to {self.distill_loss_weight}."
                    )
                    self.activated = True
                    break