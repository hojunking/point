"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .scannet import ScanNetDataset
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
)
# def process_rotation(quaternions):
#     # 3D Gaussian Splatting의 Rotation 처리 로직 (예시)
#     # Quaternion을 다른 형태로 변환하는 복잡한 로직이 올 수 있음
#     # 여기서는 단순화를 위해 첫 3차원만 반환하는 등 예시
#     # 실제 구현은 3DGS의 해당 부분 코드를 참조해야 합니다.
#     return quaternions[:, :3] # 예시: 쿼터니언의 x,y,z만 사용 (단순화)

@DATASETS.register_module()
class ScanNetDatasetBoundary(ScanNetDataset):

    def __init__(
        self,
        boundary_root=None,
        features_root=None, #
        features_flag=[],
        **kwargs,
    ):
        self.boundary_root = boundary_root
        self.features_root = features_root
        self.features_flag = features_flag

        super().__init__(**kwargs)


    def get_data(self, idx):
        # ScanNetDataset의 get_data 로직을 호출하여 기본 데이터 로드
        data_dict = super().get_data(idx)

        scene_name = self.get_data_name(idx) 

        # === BFANet: Boundary Label 로딩 로직 (boundary_root 사용) ===
        # 1. boundary_root가 설정되었는지 확인 (TypeError 방지)
        if self.boundary_root is not None:
            boundary_file_path = os.path.join(self.boundary_root, self.split, scene_name, "boundary.npy")
            
            # 2. boundary.npy 파일이 존재하는지 확인 (FileNotFoundError 방지)
            if os.path.exists(boundary_file_path):
                data_dict["boundary"] = np.load(boundary_file_path).reshape([-1]).astype(np.int32)
            else:
                logger = get_root_logger()
                logger.warning(f"Boundary label file not found at {boundary_file_path}. Initializing with zeros for {scene_name}.")
                data_dict["boundary"] = np.zeros(data_dict["coord"].shape[0], dtype=np.int32)
        # else:
        #     # boundary_root가 지정되지 않았을 때의 처리 (경고 및 0 초기화)
        #     logger = get_root_logger()
        #     logger.warning(f"boundary_root not specified. Initializing boundary labels with zeros for {scene_name}.")
        #     data_dict["boundary"] = np.zeros(data_dict["coord"].shape[0], dtype=np.int32)
        # ===============================================================
        
        if self.features_root is not None and len(self.features_flag) > 0:
            features_file_path = os.path.join(self.features_root, self.split, scene_name, "features.npy")
            if os.path.exists(features_file_path):
                all_features_3dgs = np.load(features_file_path).astype(np.float32)
                
                selected_features_list = []
                current_feature_idx_in_npy = 0 # features.npy 내 현재 특징의 시작 인덱스 (3DGS 순서)

                if "scale" in self.features_flag:
                    selected_features_list.append(all_features_3dgs[:, current_feature_idx_in_npy : current_feature_idx_in_npy + 3])
                current_feature_idx_in_npy += 3

                if "opacity" in self.features_flag:
                    selected_features_list.append(all_features_3dgs[:, current_feature_idx_in_npy : current_feature_idx_in_npy + 1])
                current_feature_idx_in_npy += 1
                
                if "rotation" in self.features_flag:
                    selected_features_list.append(all_features_3dgs[:, current_feature_idx_in_npy : current_feature_idx_in_npy + 4])
                current_feature_idx_in_npy += 4 

                if selected_features_list:
                    # 기존 data_dict["features"]를 덮어씁니다.
                    data_dict["features"] = np.concatenate(selected_features_list, axis=-1)
                else: 
                    # features_flag가 비어있지 않으나 선택된 특징이 없을 경우 (위 features_flag 유효성 검사로 방지)
                    data_dict["features"] = np.zeros((data_dict["coord"].shape[0], 0), dtype=np.float32)

            else: # features.npy 파일이 없는 경우
                logger = get_root_logger()
                logger.warning(f"Features file not found at {features_file_path}. No custom features loaded.")
                # features가 없음을 명시적으로 빈 배열로 표시합니다.
                data_dict["features"] = np.zeros((data_dict["coord"].shape[0], 0), dtype=np.float32)
        elif self.features_root is not None and len(self.features_flag) == 0:
            # features_root는 지정되었으나 features_flag가 비어있는 경우 (features를 사용하지 않음)
            logger = get_root_logger()
            logger.info(f"features_root specified but features_flag is empty. No custom features loaded for {scene_name}.")
            data_dict["features"] = np.zeros((data_dict["coord"].shape[0], 0), dtype=np.float32)
        # self.features_root가 None인 경우는 super().get_data()에서 로드된 기존 features를 유지합니다.
        # super().get_data()가 features를 로드하지 않는 경우를 대비한 방어적 초기화:
        # if "features" not in data_dict:
        #      data_dict["features"] = np.zeros((data_dict["coord"].shape[0], 0), dtype=np.float32)
        return data_dict