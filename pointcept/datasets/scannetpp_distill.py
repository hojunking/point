# In point/datasets/scannet_distill.py

import os
import numpy as np
from .builder import DATASETS
from .scannetpp import ScanNetPPDataset  # 기존 ScanNetDataset을 import
from pointcept.utils.logger import get_root_logger

@DATASETS.register_module()
class ScanNetPPDistillDataset(ScanNetPPDataset): # 혹은 ScanNetDatasetBoundary
    def __init__(self,
                 features_root=None,
                 **kwargs):
        # 부모 클래스(ScanNetDataset)의 __init__을 먼저 호출합니다.
        super().__init__(**kwargs)
        self.features_root = features_root

    def get_data(self, idx):
        # 1. 부모 클래스(ScanNetDataset)에서 기본 데이터 로드
        #    (coord, color, normal, segment 등)
        data_dict = super().get_data(idx)
        scene_name = self.get_data_name(idx)
        
        # 3. Teacher가 사용할 Opacity 특징 로드
        if self.features_root:
            try:
                features_path = os.path.join(self.features_root, self.split, scene_name, "features.npy")
                all_features_3dgs = np.load(features_path)
                # features.npy에서 opacity 데이터 추출 (scale(3), opacity(1) 순서 가정)
                data_dict['features'] = all_features_3dgs[:, 3:4].astype(np.float32)
            except FileNotFoundError:
                data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)
                logger = get_root_logger()
                logger.warning(f"Features file not found at path, {self.features_root} Filling with zeros.")
        else:
            data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)
            logger = get_root_logger()
            logger.warning(f"Features file not found at {self.features_root}. Filling with zeros.")

        return data_dict