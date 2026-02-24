_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4
num_worker = 16
mix_prob = 0.8
empty_cache = False
enable_amp = False
enable_wandb = False
seed = 43244662

# model settings
model = dict(
    type="SegmentorBSOctFormer",
    num_classes=20,
    backbone_out_channels=168,
    backbone=dict(
        type="OctFormer-v1m2-BS",
        in_channels=10,
        num_classes=20,
        fpn_channels=168,
        channels=(96, 192, 384, 384),
        num_blocks=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 24),
        patch_size=26,
        stem_down=2,
        head_up=2,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        octree_depth=11,
        octree_full_depth=2,
        bfa_head_cfg=dict(
            boundary_feature_channels=128,
            num_heads=8,
            dropout=(0.0, 0.0),
        ),
    ),
    criteria=[
        dict(
            type="BoundarySemanticLoss",
            semantic_loss_weight=1.0,
            boundary_loss_weight=1.0,
            ignore_index=-1,
            num_semantic_classes=20,
            semantic_boundary_weight_factor=9.0,
        )
    ],
)

# scheduler settings
epoch = 600
optimizer = dict(type="AdamW", lr=0.0015, weight_decay=0.05)
scheduler = dict(
    type="MultiStepWithWarmupLR",
    milestones=[0.6, 0.9],
    gamma=0.1,
    warmup_rate=0.05,
    warmup_scale=1e-5,
)
param_dicts = [dict(keyword="blocks", lr=0.00015)]

# dataset settings
dataset_type = "ScanNetDatasetBoundary"
data_root = "data/scannet"
boundary_root = "data/boundary/bfa002"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        boundary_root=boundary_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
                return_displacement=True,
                project_displacement=True,
            ),
            dict(type="SphereCrop", sample_rate=1.0, mode="random"),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "normal", "segment", "boundary"),
                feat_keys=("coord", "color", "normal", "displacement"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        boundary_root=boundary_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
                return_displacement=True,
                project_displacement=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "normal", "segment", "boundary", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal", "displacement"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        boundary_root=boundary_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="test",
                return_displacement=True,
                project_displacement=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "normal", "index", "boundary"),
                    feat_keys=("coord", "color", "normal", "displacement"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
            ],
        ),
    ),
)
