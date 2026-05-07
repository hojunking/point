# In point/datasets/scannet_distill.py

import os
import numpy as np
from .builder import DATASETS
from .scannet import ScanNetDataset  # 기존 ScanNetDataset을 import
from pointcept.utils.logger import get_root_logger


@DATASETS.register_module()
class ScanNetDistillDataset(ScanNetDataset):  # 혹은 ScanNetDatasetBoundary
    def __init__(
        self,
        features_root=None,
        features_flag=None,
        sh_degree=0,
        **kwargs,
    ):
        self.features_root = features_root
        self.features_flag = features_flag or []
        self.sh_degree = sh_degree

        debug_env = os.environ.get("SCANNET_DISTILL_DEBUG", "0").strip().lower()
        self.debug_enabled = debug_env in ("1", "true", "yes", "y", "on")
        self._debug_logged = False

        super().__init__(**kwargs)

    def get_data(self, idx):
        # ScanNetDataset의 get_data 로직을 호출하여 기본 데이터 로드
        data_dict = super().get_data(idx)

        scene_name = self.get_data_name(idx)
        features_file_path = None
        raw_feature_dim = None
        selected_feature_dim = None

        if self.features_root is not None and len(self.features_flag) > 0:
            features_file_path = os.path.join(
                self.features_root, self.split, scene_name, "features.npy"
            )
            if os.path.exists(features_file_path):
                all_features_3dgs = np.load(features_file_path).astype(np.float32)
                raw_feature_dim = int(all_features_3dgs.shape[1])

                selected_features_list = []
                current_feature_idx_in_npy = 0  # features.npy 내 현재 특징의 시작 인덱스 (3DGS 순서)

                def take_feature(dim):
                    nonlocal current_feature_idx_in_npy
                    start = current_feature_idx_in_npy
                    end = start + dim
                    current_feature_idx_in_npy = end
                    return all_features_3dgs[:, start:end]

                if "scale" in self.features_flag:
                    selected_features_list.append(take_feature(3))
                else:
                    current_feature_idx_in_npy += 3

                if "opacity" in self.features_flag:
                    selected_features_list.append(take_feature(1))
                else:
                    current_feature_idx_in_npy += 1

                if "rotation" in self.features_flag:
                    selected_features_list.append(take_feature(3))
                else:
                    current_feature_idx_in_npy += 3

                if "sh" in self.features_flag and self.sh_degree and self.sh_degree > 0:
                    sh_dim = min(48, 3 * (self.sh_degree + 1) ** 2)
                    selected_features_list.append(take_feature(sh_dim))
                    remaining = max(48 - sh_dim, 0)
                    current_feature_idx_in_npy += remaining
                else:
                    current_feature_idx_in_npy += 48

                if selected_features_list:
                    # 기존 data_dict["features"]를 덮어씁니다.
                    data_dict["features"] = np.concatenate(selected_features_list, axis=-1)
                else:
                    # features_flag가 비어있지 않으나 선택된 특징이 없을 경우
                    data_dict["features"] = np.zeros(
                        (data_dict["coord"].shape[0], 0), dtype=np.float32
                    )
                selected_feature_dim = int(data_dict["features"].shape[1])

            else:  # features.npy 파일이 없는 경우
                logger = get_root_logger()
                logger.warning(
                    f"Features file not found at {features_file_path}. No custom features loaded."
                )
                # features가 없음을 명시적으로 빈 배열로 표시합니다.
                data_dict["features"] = np.zeros(
                    (data_dict["coord"].shape[0], 0), dtype=np.float32
                )
                selected_feature_dim = 0

        elif self.features_root is not None and len(self.features_flag) == 0:
            # features_root는 지정되었으나 features_flag가 비어있는 경우 (features를 사용하지 않음)
            logger = get_root_logger()
            logger.warning(
                f"features_root specified but features_flag is empty. No custom features loaded for {scene_name}."
            )
            data_dict["features"] = np.zeros((data_dict["coord"].shape[0], 0), dtype=np.float32)
            selected_feature_dim = 0
        else:
            if "features" in data_dict:
                selected_feature_dim = int(data_dict["features"].shape[1])

        # One-shot debug log to inspect feature assembly in runtime.
        if self.debug_enabled and not self._debug_logged:
            logger = get_root_logger()
            color_dim = int(data_dict["color"].shape[1]) if "color" in data_dict else -1
            normal_dim = int(data_dict["normal"].shape[1]) if "normal" in data_dict else -1
            features_dim = int(data_dict["features"].shape[1]) if "features" in data_dict else -1
            total_feat_dim = (color_dim if color_dim > 0 else 0) + (normal_dim if normal_dim > 0 else 0) + (features_dim if features_dim > 0 else 0)
            logger.info(
                "[ScanNetDistillDataset:debug] split=%s scene=%s flags=%s sh_degree=%s features_path=%s raw_features_dim=%s selected_features_dim=%s color_dim=%d normal_dim=%d total_input_feat_dim=%d",
                self.split,
                scene_name,
                self.features_flag,
                self.sh_degree,
                features_file_path if features_file_path is not None else "N/A",
                raw_feature_dim if raw_feature_dim is not None else "N/A",
                selected_feature_dim if selected_feature_dim is not None else "N/A",
                color_dim,
                normal_dim,
                total_feat_dim,
            )
            self._debug_logged = True

        return data_dict
