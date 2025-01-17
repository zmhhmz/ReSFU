_base_ = './segmenter_vit-s_mask_8xb1-160k_ade20k-512x512.py'

model = dict(
    backbone=dict(
        type='VisionTransformerIm'
    ),
    decode_head=dict(
        _delete_=True,
        type='FCNHeadDirectGuideUpx4',
        in_channels=384,
        channels=384,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=150,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        upsample_cfg=dict(type='resfu', scale_factor=4, lr_mult=500)))
