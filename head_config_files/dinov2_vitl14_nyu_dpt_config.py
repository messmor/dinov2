dataset_type = 'NYUDataset'
data_root = '/checkpoint/dino/datasets/NYU'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (416, 544)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='NYUCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(416, 544)),
    dict(
        type='ColorAug',
        prob=0.5,
        gamma_range=[0.9, 1.1],
        brightness_range=[0.75, 1.25],
        color_range=[0.9, 1.1]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'depth_gt'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg', 'cam_intrinsic'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 640),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'cam_intrinsic'))
        ])
]
eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg', 'cam_intrinsic'))
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='NYUDataset',
        data_root='/checkpoint/dino/datasets/NYU',
        depth_scale=1000,
        split='nyu_train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='DepthLoadAnnotations'),
            dict(type='NYUCrop', depth=True),
            dict(type='RandomRotate', prob=0.5, degree=2.5),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomCrop', crop_size=(416, 544)),
            dict(
                type='ColorAug',
                prob=0.5,
                gamma_range=[0.9, 1.1],
                brightness_range=[0.75, 1.25],
                color_range=[0.9, 1.1]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'depth_gt'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'cam_intrinsic'))
        ],
        garg_crop=False,
        eigen_crop=True,
        min_depth=0.001,
        max_depth=10),
    val=[
        dict(
            type='NYUDataset',
            data_root='/checkpoint/dino/datasets/NYU',
            depth_scale=1000,
            split='nyu_test.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(480, 640),
                    flip=True,
                    flip_direction='horizontal',
                    transforms=[
                        dict(type='RandomFlip', direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(
                            type='Collect',
                            keys=['img'],
                            meta_keys=('filename', 'ori_filename', 'ori_shape',
                                       'img_shape', 'pad_shape',
                                       'scale_factor', 'flip',
                                       'flip_direction', 'img_norm_cfg',
                                       'cam_intrinsic'))
                    ])
            ],
            garg_crop=False,
            eigen_crop=True,
            min_depth=0.001,
            max_depth=10)
    ],
    test=[
        dict(
            type='NYUDataset',
            data_root='/checkpoint/dino/datasets/NYU',
            depth_scale=1000,
            split='nyu_test.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(480, 640),
                    flip=True,
                    flip_direction='horizontal',
                    transforms=[
                        dict(type='RandomFlip', direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(
                            type='Collect',
                            keys=['img'],
                            meta_keys=('filename', 'ori_filename', 'ori_shape',
                                       'img_shape', 'pad_shape',
                                       'scale_factor', 'flip',
                                       'flip_direction', 'img_norm_cfg',
                                       'cam_intrinsic'))
                    ])
            ],
            garg_crop=False,
            eigen_crop=True,
            min_depth=0.001,
            max_depth=10),
        dict(
            type='SUNRGBDDataset',
            data_root='/checkpoint/dino/datasets/eval/SUNRGBD',
            depth_scale=8000,
            split='SUNRGBD_val_splits.txt',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(480, 640),
                    flip=True,
                    flip_direction='horizontal',
                    transforms=[
                        dict(
                            type='Resize',
                            img_scale=(480, 640),
                            keep_ratio=False),
                        dict(type='RandomFlip', direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ],
            min_depth=0.001,
            max_depth=10)
    ])
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = ''
workflow = [('train', 1)]
cudnn_benchmark = True
model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='DinoVisionTransformer',
        final_norm=False,
        with_cls_token=True,
        output_cls_token=True,
        frozen_stages=100,
        out_indices=[4, 11, 17, 23]),
    decode_head=dict(
        type='DPTHead',
        norm_cfg=None,
        min_depth=0.001,
        max_depth=10,
        loss_decode=[
            dict(
                type='SigLoss',
                valid_mask=True,
                loss_weight=1.0,
                warm_up=True,
                loss_name='loss_depth'),
            dict(
                type='GradientLoss',
                valid_mask=True,
                loss_weight=0.5,
                loss_name='loss_grad')
        ],
        in_channels=[1024, 1024, 1024, 1024],
        channels=256,
        embed_dims=1024,
        post_process_channels=[128, 256, 512, 1024],
        readout_type='project'),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
max_lr = 0.0001
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_embed=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=12800,
    warmup_ratio=0.001,
    min_lr_ratio=1e-08,
    by_epoch=False)
momentum_config = dict(policy='OneCycle')
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='IterBasedRunner', max_iters=38400)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(
    by_epoch=False,
    interval=800,
    pre_eval=True,
    rule='less',
    save_best='abs_rel',
    greater_keys=('a1', 'a2', 'a3'),
    less_keys=('abs_rel', 'rmse'))
work_dir = '/checkpoint/dino/evaluations/depth/dinov2_vitl14_nyu_dpt'
gpu_ids = range(0, 1)
