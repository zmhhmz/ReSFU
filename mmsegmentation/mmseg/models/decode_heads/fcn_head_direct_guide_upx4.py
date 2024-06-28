# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from ..upsamplers import build_upsampler


@MODELS.register_module()
class FCNHeadDirectGuideUpx4(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 upsample_cfg=dict(
                     type='resfu'
                 ),
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input and num_convs>0:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg)
        elif self.concat_input and num_convs==0:
            self.conv_cat = ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        self.shallow_feat_extractor = nn.Sequential(nn.Conv2d(3, 64, 3,padding=1,stride=2, padding_mode='replicate'),
                                   nn.GroupNorm(1,64),
                                   nn.ReLU(True),
                                   nn.Conv2d(64,32, 3,padding=1,stride=2, padding_mode='replicate')
                                   )
        self.upsampler = build_upsampler(self.upsample_cfg, in_channels=self.channels, scale_factor=4, guide_channels=32)

    def _forward_feature(self, im, x):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        shallow_feat = self.shallow_feat_extractor(im)
        feats = self.convs(x)
        if self.concat_input and self.num_convs>0:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        if self.concat_input and self.num_convs==0:
            feats = self.conv_cat(feats)
        feats = self.upsampler(shallow_feat, feats)
        return feats

    def forward(self, inputs):
        """Forward function."""
        im, x = inputs
        output = self._forward_feature(im, x)
        output = self.cls_seg(output)
        return output
