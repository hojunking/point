# semseg-pt-v3m4-distill.py
_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# misc custom setting
batch_size = 4
num_worker = 12
mix_prob = 0.8  # mix_prob는 증류 학습 시에는 0으로 설정하는 것을 고려해볼 수 있습니다.
empty_cache = False
enable_amp = True
enable_wandb = False

# --- 여기에 새로운 플래그 추가 ---
enable_distillation = True  # 증류를 사용하려면 True, 끄려면 False

# model settings
model = dict(
    type="SegmentorBSDistill",
    num_classes=100,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m4",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),

        bsblock_cfg=dict(
            in_channels=64,  # PTv3 dec_channels[0]과 일치 
            semantic_out_channels=64, # DefaultSegmentorBS의 backbone_out_channels와 일치 
            boundary_feature_channels=128, # BFANet 논문 참조 
            num_heads=8, 
            dropout=[0.0, 0.0],
            num_semantic_classes=100,
        ),
    ),
    teacher_backbone=dict(
        type="PT-v3m2",
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        #checkpoint_path="pre_trained/sonata_opacity_k20_400epoch.pth",
        checkpoint_path="exp/scannetpp/sonata_opacity_k20_200epoch_scannetpp_dl4/model/model_last.pth",
        #checkpoint_path="pre_trained/sonata_color-opacity_m.pth",
    ),
    mlp_bridge=dict(
        # Student와 Teacher 인코더의 마지막 출력 채널과 일치시킵니다.
        in_channels=512,
        out_channels=512,
        hidden_channels=256
    ),
    
    criteria=[dict(
            type="BSLossWithLovasz", # [수정] 새로운 클래스 이름
            semantic_loss_weight=1.0,
            boundary_loss_weight=0.9,
            lovasz_loss_weight=1.0,   # [추가] Lovasz 손실 가중치
            ignore_index=-1,
            num_semantic_classes=20,
            semantic_boundary_weight_factor=9.0
        ),
        # 보조 증류 손실
        dict(type="FeatureDistillationLoss", loss_weight=0.0, loss_type="CosineSimilarity")
            # CosineSimilarity L2
    ],
)

# scheduler settings
epoch = 800
#optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    #max_lr=[0.006, 0.0006],
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
#param_dicts = [dict(keyword="block", lr=0.0006)]
param_dicts = [dict(keyword="block", lr=0.0001)]

# dataset settings
dataset_type = "ScanNetPPBSDistillDataset"
data_root = "data/scannetpp"

features_root = 'data/features/scannetpp_mah_k20'
boundary_root = 'data/boundary/scannetpp_scale07_bfa004'
data = dict(
    num_classes=100,
    ignore_index=-1,

    train=dict(
        type=dataset_type,
        split="train_grid1mm_chunk6x6_stride3x3",
        data_root=data_root,
        features_root=features_root,
        boundary_root=boundary_root,
        # lr_file="data/scannet/train100_samples.txt",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "boundary"),
                feat_keys=("color", "normal"),
                feat_teacher_keys=("color", "features")
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        features_root=features_root,
        boundary_root=boundary_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse", "boundary"),
                feat_keys=("color", "normal"),
                feat_teacher_keys=("color", "features")
            ),
        ],
        test_mode=False,
    ),
    
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        features_root=features_root,
        boundary_root=boundary_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "boundary"),
                    #feat_keys=("color", "normal"),
                    feat_keys=("color", "normal"),
                    feat_teacher_keys=("color", "features")
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),          # 학습 중 검증(validation) 담당
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False), # [핵심] 학습 후 최종 테스트(test) 담당
]

if enable_distillation:
    hooks.append(
        dict(type="DistillationSchedulerHook",
             start_epoch=0, 
             distill_loss_weight=0.4)
    )