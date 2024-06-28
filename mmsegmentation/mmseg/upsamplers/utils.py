import torch
import torch.nn as nn
import torch.nn.functional as F

def create_gaussian_kernel(kernel_size, sigma=1):
    # Create a 2D Gaussian kernel for generating q_GS
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    y = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(x, y)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel

class GuidedFilter(nn.Module):
    def __init__(self, radius, epsilon, channels, scale=1):
        super(GuidedFilter, self).__init__()
        self.radius = radius//scale
        self.epsilon = epsilon
        self.scale=scale
        kernel = torch.ones(channels, 1, 1, 2 * self.radius + 1)
        self.register_buffer('kernel', kernel)

    def forward(self, I, p):
        
        if self.scale>1:
            I0 = I.clone()
            I = F.interpolate(I, scale_factor=1./self.scale, mode="nearest")
            p = F.interpolate(p, scale_factor=1./self.scale, mode="nearest")
        
        N = self.box_filter_separable(torch.ones_like(I))

        mean_I = self.box_filter_separable(I) / N
        mean_p = self.box_filter_separable(p) / N
        mean_Ip = self.box_filter_separable(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.box_filter_separable(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.epsilon)
        b = mean_p - a * mean_I

        mean_a = self.box_filter_separable(a) / N
        mean_b = self.box_filter_separable(b) / N
        
        if self.scale>1:
            mean_a = F.interpolate(mean_a, I0.shape[-2:], mode='bilinear')
            mean_b = F.interpolate(mean_b, I0.shape[-2:], mode='bilinear')
            q = mean_a * I0 + mean_b
        else:
            q = mean_a * I + mean_b
        return q


    def box_filter_separable(self, x):
        x = F.conv2d(x, self.kernel, padding=(0, self.radius), groups=x.size(1))
        x = F.conv2d(x, self.kernel.transpose(2, 3), padding=(self.radius, 0), groups=x.size(1))
        return x


