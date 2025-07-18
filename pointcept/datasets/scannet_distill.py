# In point/datasets/scannet_distill.py

import os
import numpy as np
from .builder import DATASETS
from .scannet import ScanNetDataset  # 기존 ScanNetDataset을 import

@DATASETS.register_module()
class ScanNetDistillDataset(ScanNetDataset): # 혹은 ScanNetDatasetBoundary
    def __init__(self, features_root=None, **kwargs):
        super().__init__(**kwargs)
        self.features_root = features_root

    def get_data(self, idx):
        # 1. 부모 클래스에서 기본 데이터(coord, color, normal, segment) 로드
        data_dict = super().get_data(idx)
        
        # 2. Teacher가 사용할 추가 특징(opacity) 로드
        if self.features_root:
            try:
                # features.npy 파일에 [scale(3), opacity(1), ...] 순서라고 가정
                features_path = os.path.join(self.features_root, self.get_data_name(idx), "features.npy")
                all_features_3dgs = np.load(features_path)
                data_dict['features'] = all_features_3dgs[:, 3:4].astype(np.float32)
            except FileNotFoundError:
                data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)
        else:
             data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)
        
        # 'feat'나 'feat_teacher'를 여기서 만들지 않고 원본 데이터를 그대로 반환
        return data_dict