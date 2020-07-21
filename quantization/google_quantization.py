import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn import Parameter

import numpy as np
import os
import matplotlib.pyplot as plt

# global params
# QUANTIZE_NUM = 127.0


def quantization_on_weights(x, k):
    n = 2 ** k
    a = torch.min(x)
    b = torch.max(x)
    s = (b - a) / (n - 1)

    x = torch.clamp(x, float(a), float(b))
    x = (x - a) / s
    x = RoundFunction.apply(x)
    x = x * s + a
    return x


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantization_on_input(x, k):
    n = 2 ** k
    a = torch.min(x)
    b = torch.max(x)
    s = (b - a) / (n - 1)

    x = torch.clamp(x, float(a), float(b))
    x = (x - a) / s
    x = RoundFunction.apply(x)
    x = x * s + a
    return x


class QConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits_weights=32, bits_activations=32):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                               stride, padding, dilation, groups, bias)

        self.bits_weights = bits_weights
        self.bits_activations = bits_activations


    def forward(self, input):
        if self.bits_activations == 32:
            quantized_input = input
        else:
            quantized_input = quantization_on_input(input, self.bits_activations)
        quantized_weight = quantization_on_weights(self.weight, self.bits_weights)
        return F.conv2d(quantized_input, quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("google")
        return s


class QLinear(nn.Linear):
    """
    custom linear layers for quantization
    """
    def __init__(self, in_features, out_features, bias=False, bits_weights=32, bits_activations=32):
        super(QLinear, self).__init__(in_features, out_features, bias=bias)
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, input):
        quantized_input = quantization_on_input(input, self.bits_activations)
        quantized_weight = quantization_on_weights(self.weight, self.bits_weights)
        return F.linear(quantized_input, quantized_weight, self.bias)

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("google")
        return s

if __name__ == "__main__":
    x = torch.rand(2,2,2,2)
    print('x={}'.format(x))
    k = 8
    q_x = quantization_on_weights(x, k)
    print('q_x={}'.format(q_x))