import torch
import torch.nn as nn
from resfu import ReSFU

# Non-hierarchical network architecutre, e.g., Segmenter;
# direct 4x upsampling of deep LR feature x under the guidance of shallow HR feature y
h,w = 32,32
up_ratio = 4
H,W = up_ratio*h,up_ratio*w
B=2
C,c = 128,32
y = torch.randn(B,c,H,W).to('cuda') # shallow guidance HR feature
x = torch.randn(B,C,h,w).to('cuda')  # deep LR feature
ReSFU_module = ReSFU(dim_y=c,dim_x=C,scale_factor=4).to('cuda')
x_up = ReSFU_module(y,x)
print('====================Non-hierarchical====================')
print('The LR feature x({0}) is upsampled under the guidance of y({1}) to x_up({2}).'.format(x.size(),y.size(),x_up.size()))

# Hierarchical network architecutre, e.g., SegFormer;
# direct upsampling of features from different stages:
# c4=(8x)=>c4_up, c3=(4x)=>c3_up, c2=(2x)=>c2_up, under the guidance of c1
H,W = 128,128
B = 2
C = 32
c1 = torch.randn(B,C,H,W).to('cuda')
c2 = torch.randn(B,C,H//2,W//2).to('cuda')
c3 = torch.randn(B,C,H//4,W//4).to('cuda')
c4 = torch.randn(B,C,H//8,W//8).to('cuda')
ReSFU_modules = nn.ModuleList()
for i in range(3):
    ReSFU_modules.append(ReSFU(dim_y=C,dim_x=C,scale_factor=2**(i+1)).to('cuda'))

c2_up = ReSFU_modules[0](c1, c2)
c3_up = ReSFU_modules[1](c1, c3)
c4_up = ReSFU_modules[2](c1, c4)
print('======================Hierarchical======================')
print('Under the guidance of early stage feature c1({0}), the LR featurs c2({1}),\
 c3({2}) and c4({3}) are upsampled to c2_up({4}), c3_up({5}) and c4_up({6}), respectively.'\
        .format(c1.size(), c2.size(), c3.size(), c4.size(), c2_up.size(), c3_up.size(), c4_up.size()))