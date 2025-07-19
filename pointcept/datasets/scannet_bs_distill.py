# In point/datasets/scannet_bs_distill.py

import os
import numpy as np
from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .scannet import ScanNetDataset  # [수정] 기본 ScanNetDataset을 import하여 상속

@DATASETS.register_module()
class ScanNetBSDistillDataset(ScanNetDataset):
    def __init__(self,
                 boundary_root=None,
                 features_root=None,
                 **kwargs):
        # 부모 클래스(ScanNetDataset)의 __init__을 먼저 호출합니다.
        super().__init__(**kwargs)
        self.boundary_root = boundary_root
        self.features_root = features_root

    def get_data(self, idx):
        # 1. 부모 클래스(ScanNetDataset)에서 기본 데이터 로드
        #    (coord, color, normal, segment 등)
        data_dict = super().get_data(idx)
        scene_name = self.get_data_name(idx)
        
        # 2. Boundary 레이블 로드
        if self.boundary_root:
            try:
                boundary_path = os.path.join(self.boundary_root, self.split, scene_name, "boundary.npy")
                data_dict['boundary'] = np.load(boundary_path).reshape(-1).astype(np.int32)
            except FileNotFoundError:
                logger = get_root_logger()
                logger.warning(f"Boundary file not found at {boundary_path}. Filling with zeros.")
                data_dict['boundary'] = np.zeros(data_dict['coord'].shape[0], dtype=np.int32)
        else:
            # boundary_root가 config에 없으면 0으로 채움
            data_dict['boundary'] = np.zeros(data_dict['coord'].shape[0], dtype=np.int32)

        # 3. Teacher가 사용할 Opacity 특징 로드
        if self.features_root:
            try:
                features_path = os.path.join(self.features_root, scene_name, "features.npy")
                all_features_3dgs = np.load(features_path)
                # features.npy에서 opacity 데이터 추출 (scale(3), opacity(1) 순서 가정)
                data_dict['features'] = all_features_3dgs[:, 3:4].astype(np.float32)
            except FileNotFoundError:
                data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)
        else:
            data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)
        
        return data_dict