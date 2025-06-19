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


@DATASETS.register_module()
class ScanNetDatasetBoundary(ScanNetDataset):

    def __init__(
        self,
        boundary_root=None,
        **kwargs,
    ):
        self.boundary_root = boundary_root
        
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
        else:
            # boundary_root가 지정되지 않았을 때의 처리 (경고 및 0 초기화)
            logger = get_root_logger()
            logger.warning(f"boundary_root not specified. Initializing boundary labels with zeros for {scene_name}.")
            data_dict["boundary"] = np.zeros(data_dict["coord"].shape[0], dtype=np.int32)
        # ===============================================================
            
        return data_dict