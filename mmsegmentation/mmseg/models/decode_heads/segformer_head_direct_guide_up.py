# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from ..upsamplers import build_upsampler


@MODELS.register_module()
class SegformerHeadDirectGuideUp(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """
    def __init__(self, interpolate_mode='bilinear',  
                 upsample_cfg=dict(type='resfu', guided=True), **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.upsample_cfg = upsample_cfg.copy()
        self.guided_upsample = upsample_cfg['guided']
        self.upsample_modules = nn.ModuleList()
        
        self.convs = nn.ModuleList()
        scale_factors=[2,4,8]
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            if i>0:
                self.upsample_modules.append(build_upsampler(self.upsample_cfg, in_channels=self.channels,
                                                            scale_factor=scale_factors[i-1],guide_channels=self.channels))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)


    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            conv = self.convs[idx]
            x = conv(inputs[idx])
            if idx==0:
                outs.append(x)
            else:
                if self.guided_upsample:
                    outs.append(self.upsample_modules[idx-1](outs[0], x))
                else:
                    outs.append(self.upsample_modules[idx-1](x))
                    
        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out
