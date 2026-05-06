"""
SONATA pretraining with PUPGS opacity as the only custom 3DGS feature.

This config keeps the base SONATA schedule/model settings, but fixes the
input-feature contract at config construction time:
  global/local feat = coord(3) + color(3) + opacity(1)
"""

_base_ = ["pretrain-sonata-v1m1-0-base.py"]

model = dict(backbone=dict(in_channels=7))

features_root = "data/features/mah_k10_pupgs"
feature_flag = ["opacity"]
sh_degree = 0

view_keys = ("coord", "origin_coord", "color", "features")
global_feat_keys = ("global_coord", "global_color", "global_features")
local_feat_keys = ("local_coord", "local_color", "local_features")

transform = [
    dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=view_keys,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        max_size=65536,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.02}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_color",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_color",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=global_feat_keys,
        local_feat_keys=local_feat_keys,
    ),
]

data = dict(
    train=dict(
        type="ScanNetDistillDataset",
        features_root=features_root,
        features_flag=feature_flag,
        sh_degree=sh_degree,
        transform=transform,
    )
)
