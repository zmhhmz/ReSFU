import torch.nn as nn
from .resfu import ReSFU


def build_upsampler(cfg, in_channels, scale_factor, guide_channels=None):
    upsample_mode = cfg['type']
    if upsample_mode == 'nearest':
        return nn.Upsample(scale_factor=scale_factor, mode='nearest')
    elif upsample_mode == 'bilinear':
        bilinear_cfg = dict(align_corners=False)
        bilinear_cfg.update(cfg)
        return nn.Upsample(scale_factor=scale_factor,
                           mode='bilinear',
                           align_corners=bilinear_cfg['align_corners'])
    elif upsample_mode == 'deconv':
        deconv_cfg = dict(kernel_size=3,
                          stride=2,
                          padding=1,
                          output_padding=1)
        deconv_cfg.update(cfg)
        return nn.ConvTranspose2d(in_channels,
                                  in_channels,
                                  kernel_size=deconv_cfg['kernel_size'],
                                  stride=deconv_cfg['stride'],
                                  padding=deconv_cfg['padding'],
                                  output_padding=deconv_cfg['output_padding'])
    elif upsample_mode == 'pixelshuffle':
        pixelshuffle_cfg = dict(kernel_size=3,
                                padding=1)
        pixelshuffle_cfg.update(cfg)
        return nn.Sequential(nn.Conv2d(in_channels,
                                       in_channels * scale_factor ** 2,
                                       kernel_size=pixelshuffle_cfg['kernel_size'],
                                       padding=pixelshuffle_cfg['padding']),
                             nn.PixelShuffle(upscale_factor=scale_factor))
    elif upsample_mode == 'resfu':
        assert guide_channels is not None
        resfu_cfg = dict(groups=4,
                        kernel_size=3,
                        scale_factor=2,
                        embedding_dim=32,
                        y_conv=True, qkv_bias=True,
                        normx=True, normy=True,
                        gn_dim=8, radius=8,
                        eps=0.001, lr_mult=1., gf_scale=2
                        )
        resfu_cfg.update(cfg)
        return ReSFU(guide_channels, in_channels,
                     groups=resfu_cfg['groups'],
                     kernel_size=resfu_cfg['kernel_size'],
                     scale_factor=resfu_cfg['scale_factor'],
                     embedding_dim=resfu_cfg['embedding_dim'],
                     y_conv=resfu_cfg['y_conv'],
                     qkv_bias=resfu_cfg['qkv_bias'],
                     normx=resfu_cfg['normx'],
                     normy=resfu_cfg['normy'],
                     gn_dim=resfu_cfg['gn_dim'],
                     radius=resfu_cfg['radius'],
                     eps=resfu_cfg['eps'],
                     lr_mult=resfu_cfg['lr_mult'],
                     gf_scale=resfu_cfg['gf_scale'])
    else:
        raise NotImplementedError

