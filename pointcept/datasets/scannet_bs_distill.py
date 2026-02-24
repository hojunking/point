# In point/datasets/scannet_bs_distill.py

import os
import numpy as np
from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .scannet import ScanNetDataset  # [수정] 기본 ScanNetDataset을 import하여 상속
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)
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
        logger = get_root_logger()
        if self.boundary_root:
            try:
                boundary_path = os.path.join(self.boundary_root, self.split, scene_name, "boundary.npy")
                data_dict['boundary'] = np.load(boundary_path).reshape(-1).astype(np.int32)
            except FileNotFoundError:
                logger.warning(f"Boundary file not found at {boundary_path}. Filling with zeros.")
                data_dict['boundary'] = np.zeros(data_dict['coord'].shape[0], dtype=np.int32)
        else:
            # boundary_root가 config에 없으면 0으로 채움 + 경고
            data_dict['boundary'] = np.zeros(data_dict['coord'].shape[0], dtype=np.int32)
            logger.warning("boundary_root is empty. Filling boundary labels with zeros.")

        # 3. Teacher가 사용할 Opacity 특징 로드
        if self.features_root:
            try:
                features_path = os.path.join(self.features_root, self.split, scene_name, "features.npy")
                all_features_3dgs = np.load(features_path)
                # features.npy에서 opacity 데이터 추출 (scale(3), opacity(1) 순서 가정)
                data_dict['features'] = all_features_3dgs[:, 3:4].astype(np.float32)
            except FileNotFoundError:
                data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)
                logger.warning(f"Features file not found at path, {self.features_root} Filling with zeros.")
        else:
            data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)
            logger.warning(f"Features file not found at {self.features_root}. Filling with zeros.")

        return data_dict

@DATASETS.register_module()
class ScanNet200DatasetBSDistill(ScanNetDataset): # <--- ScanNetDataset을 직접 상속
    # 200 클래스에 맞는 속성들을 여기서 직접 오버라이드합니다.
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200", # 200 클래스 세그먼트 파일을 사용하도록 지정
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_200) # 200 클래스 맵을 사용하도록 지정

    def __init__(self,
                 boundary_root=None,
                 features_root=None,
                 **kwargs):
        # 부모 클래스(ScanNetDataset)의 __init__을 먼저 호출합니다.
        super().__init__(**kwargs)
        self.boundary_root = boundary_root
        self.features_root = features_root
        
        # VALID_ASSETS에 boundary와 features도 추가해줍니다.
        self.VALID_ASSETS.append("boundary")
        self.VALID_ASSETS.append("features")

    def get_data(self, idx):
        # 1. 부모 클래스(ScanNetDataset)에서 기본 데이터 로드
        #    (coord, color, normal, 그리고 segment200을 segment 키로)
        data_dict = super().get_data(idx)
        scene_name = self.get_data_name(idx)
        logger = get_root_logger()

        # 2. Boundary 레이블 로드 (기존 로직과 동일)
        if self.boundary_root:
            try:
                boundary_path = os.path.join(self.boundary_root, self.split, scene_name, "boundary.npy")
                data_dict['boundary'] = np.load(boundary_path).reshape(-1).astype(np.int32)
            except FileNotFoundError:
                logger.warning(f"Boundary file not found at {boundary_path}. Filling with zeros.")
                data_dict['boundary'] = np.zeros(data_dict['coord'].shape[0], dtype=np.int32)
        else:
            data_dict['boundary'] = np.zeros(data_dict['coord'].shape[0], dtype=np.int32)
            logger.warning("boundary_root is empty. Filling boundary labels with zeros.")

        # 3. Teacher가 사용할 Opacity 특징 로드 (기존 로직과 동일)
        if self.features_root:
            try:
                features_path = os.path.join(self.features_root, self.split, scene_name, "features.npy")
                all_features_3dgs = np.load(features_path)
                data_dict['features'] = all_features_3dgs[:, 3:4].astype(np.float32)
            except FileNotFoundError:
                data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)
                logger.warning(f"Features file not found at {features_path}. Filling with zeros.")
        else:
            data_dict['features'] = np.zeros((data_dict['coord'].shape[0], 1), dtype=np.float32)

        return data_dict
