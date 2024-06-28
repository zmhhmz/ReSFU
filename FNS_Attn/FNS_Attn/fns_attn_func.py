import torch
import fns_attn_ext
from torch.autograd import Function


class FNSAttnFunction(Function):

    @staticmethod
    def forward(ctx, attn, value, kernel_size):
        assert attn.size(2) == kernel_size * kernel_size
        # assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size

        H, W = attn.size(2), attn.size(3)
        n, c, h, w = value.size()
        output = value.new_zeros((n, c, H, W))
        if value.is_cuda:
            fns_attn_ext.forward(value, attn, kernel_size, output)
        else:
            raise NotImplementedError

        if attn.requires_grad or value.requires_grad:
            ctx.save_for_backward(attn, value)
        return output
 
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        attn, value = ctx.saved_tensors
        kernel_size = ctx.kernel_size

        grad_attn = torch.zeros_like(attn)
        grad_value = torch.zeros_like(value)
        fns_attn_ext.backward(grad_output.contiguous(), value, attn,
                        kernel_size, grad_value, grad_attn)

        return grad_attn, grad_value, None


fns_attn = FNSAttnFunction.apply

