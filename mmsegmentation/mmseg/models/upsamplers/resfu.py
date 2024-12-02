import torch
import torch.nn as nn
import torch.nn.functional as F
from FNS_Attn import fns_attn
from .utils import create_gaussian_kernel, GuidedFilter


class PCDC_Block(nn.Module):
    def __init__(self,in_channels,scale_factor,mid_channels=128,kernel_size=3,groups=4,gn_dim=8):
        super(PCDC_Block,self).__init__() 
        out_channels=kernel_size**2
        self.dilation=scale_factor
        self.groups=groups
        self.norm=nn.GroupNorm(gn_dim,in_channels)
        self.channel_compressor=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(mid_channels,mid_channels,1,groups=groups),
            nn.ReLU(True),
            nn.GroupNorm(gn_dim,mid_channels),
            nn.Conv2d(mid_channels,out_channels, 1)
        )
        self.padding=(self.dilation*(kernel_size-1))//2
        self.weight=nn.Parameter(torch.Tensor(mid_channels, in_channels//groups, kernel_size, kernel_size))
        self.bias=nn.Parameter(torch.Tensor(1,mid_channels,1,1))
        nn.init.kaiming_normal_(self.weight,mode='fan_out',nonlinearity='relu')
        torch.nn.init.zeros_(self.bias)

    def forward(self, k, q):
        # GroupNorm with shared affine weights
        q=self.norm(q)
        k=self.norm(k)
        # PCDC layer
        k=F.pad(k,(self.padding,self.padding,self.padding,self.padding),mode='replicate')
        k=F.conv2d(k,self.weight,dilation=self.dilation,groups=self.groups)
        kernel_diff=self.weight.sum(-2,keepdim=True).sum(-1,keepdim=True)
        q=F.conv2d(q,kernel_diff,groups=self.groups)
        v=k-q+self.bias
        # channel compressor
        return self.channel_compressor(v)


class ReSFU(nn.Module):
    def __init__(self, dim_y, dim_x=None, groups=4, kernel_size=3, scale_factor=4, embedding_dim=32,
                 y_conv=True, qkv_bias=True, normx=True, normy=True, 
                 gn_dim=8, radius=8, eps=0.001, lr_mult=1., gf_scale=2):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y
        self.dim_x = dim_x
        self.scale_factor = scale_factor

        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim
        self.normx = normx
        self.normy = normy

        if self.normx:
            self.norm_x = nn.GroupNorm(32, dim_x)

        if self.normy:
            self.norm_y = nn.GroupNorm(32, dim_y)

        self.GF = GuidedFilter(radius,eps,embedding_dim,scale=gf_scale)

        if y_conv:
            self.q = nn.Conv2d(dim_y, embedding_dim, kernel_size=1, bias=qkv_bias)
        else:
            assert dim_y==embedding_dim
            self.q = nn.Identity()
        self.k = nn.Conv2d(dim_x, embedding_dim, kernel_size=1, bias=qkv_bias)

        self.blur_kernels3 = create_gaussian_kernel(3).view(1, 1, 3, 3).cuda()

        self.semantic_scores = PCDC_Block(embedding_dim, scale_factor, kernel_size=kernel_size, groups=groups, gn_dim=gn_dim)
        self.detail_scores = PCDC_Block(embedding_dim, scale_factor, kernel_size=kernel_size, groups=groups, gn_dim=gn_dim)
        for name, param in self.semantic_scores.named_parameters():
            param.register_hook(lambda grad: grad * lr_mult)
        for name, param in self.detail_scores.named_parameters():
            param.register_hook(lambda grad: grad * lr_mult)
        self.apply(self._init_weights)

    def forward(self, y, x):
        bs = x.size(0)
        
        if self.normx:
            x_ = self.norm_x(x)
        else:
            x_ = x

        if self.normy:
            y = self.norm_y(y)

        q = self.q(y)
        k = self.k(x_)
        k_ = F.interpolate(k, size=q.size()[-2:], mode='bilinear')

        q_gs = F.pad(q, (1, 1, 1, 1), mode='replicate')
        q_gs = F.conv2d(q_gs, self.blur_kernels3.repeat(q.shape[1],1,1,1), groups=q.shape[1])

        q_gf = self.GF(q, k_)

        semantic_scores = self.semantic_scores(k_, q_gf)
        detail_scores = self.detail_scores(q_gs, q) 

        scores = semantic_scores+detail_scores
        # for pixels out of range, remove their influences to the softmax values
        zeros = scores.new_zeros(bs, 1, *q.size()[-2:])
        pad_num = (self.scale_factor*(self.kernel_size-1))//2
        zeros = F.pad(zeros, (pad_num, pad_num, pad_num, pad_num), value=-1e7)  
        zeros = F.unfold(zeros, kernel_size=self.kernel_size, padding=0, dilation=self.scale_factor).view(*scores.size())
        attn = F.softmax(scores+zeros, dim=1)

        out = fns_attn(attn.contiguous(), x.contiguous(), self.kernel_size)

        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
