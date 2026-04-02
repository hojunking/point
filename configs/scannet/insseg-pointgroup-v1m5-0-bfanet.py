_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 2
num_worker = 16
mix_prob = 0
empty_cache = False
enable_amp = True
evaluate = True
enable_wandb = False
seed = 43244662

class_names = [
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
]
num_classes = 20
segment_ignore_index = (-1, 0, 1)

model = dict(
    type="PG-v1m5",
    backbone=dict(
        type="OctFormer-v1m3-BS-InsSeg",
        in_channels=10,
        num_classes=num_classes,
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
    backbone_out_channels=168,
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=300,
    cluster_propose_points=100,
    cluster_min_points=50,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=-1,
        ),
    ],
    criteria_bs=[
        dict(
            type="BoundarySemanticLoss",
            semantic_loss_weight=1.0,
            boundary_loss_weight=1.0,
            ignore_index=-1,
            num_semantic_classes=num_classes,
            semantic_boundary_weight_factor=9.0,
        )
    ],
)

epoch = 800
optimizer = dict(type="AdamW", lr=0.004, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.004, 0.0004],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0004)]

dataset_type = "ScanNetDatasetBoundary"
data_root = "data/scannet"
boundary_root = "data/boundary/bfa002"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
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
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "normal",
                    "segment",
                    "boundary",
                    "instance",
                    "instance_centroid",
                    "bbox",
                ),
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
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
                return_displacement=True,
                project_displacement=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "normal",
                    "segment",
                    "boundary",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                ),
                feat_keys=("coord", "color", "normal", "displacement"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
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
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
                return_displacement=True,
                project_displacement=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "normal",
                    "segment",
                    "boundary",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "name",
                ),
                feat_keys=("coord", "color", "normal", "displacement"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
    ),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test = dict(
    type="InsSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=False,
)
