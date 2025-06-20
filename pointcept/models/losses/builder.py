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
    
def build_criteria(cfg):
    return Criteria(cfg)

def build_criteria_bs(cfg):
    return Criteria_bs(cfg)
    
