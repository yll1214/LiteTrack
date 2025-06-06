import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


# Helper functions to convert arguments to n-tuples
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerScale(nn.Module):
    """Layer scale module that scales inputs by learnable parameter gamma"""

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace  # Whether to perform operation in-place
        self.gamma = nn.Parameter(init_values * torch.ones(dim))  # Learnable scaling parameter

    def forward(self, x):
        """Scale input tensor by gamma parameter"""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob  # Probability of dropping a path
        self.scale_by_keep = scale_by_keep  # Whether to scale output by keep probability

    def forward(self, x):
        """Forward pass with drop path"""
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        """Extra representation string for module"""
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class LinearDWConv(nn.Module):
    """Linear depthwise convolution implemented with linear layer + GELU"""

    def __init__(self, hidden_features):
        super(LinearDWConv, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.GELU()
        )

    def forward(self, x):
        """Forward pass through linear depthwise convolution"""
        return self.linear(x)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # Default output features to input features
        hidden_features = hidden_features or in_features  # Default hidden features to input features
        bias = to_2tuple(bias)  # Convert bias to tuple
        drop_probs = to_2tuple(drop)  # Convert drop probabilities to tuple
        self.dwconv = LinearDWConv(hidden_features=hidden_features)

        # First linear layer: linear transform + activation + dropout
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()  # Activation function (default GELU)
        self.drop1 = nn.Dropout(drop_probs[0])  # First dropout layer

        # Second linear layer: linear transform + dropout
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])  # Second dropout layer

    def forward(self, x):
        """Forward pass through MLP"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        # Optional: depthwise convolution with residual connection
        # x_dwconv = self.dwconv(x)
        # x = x + x_dwconv

        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x  # Return original input if drop_prob is 0 or not in training mode
    keep_prob = 1 - drop_prob  # Calculate keep probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Shape for broadcasting
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)  # Random tensor with keep probability
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)  # Scale by keep probability if enabled
    return x * random_tensor  # Apply drop path


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, freeze_bn=False):
    """
    Construct convolution layer with optional frozen batch norm
    """
    if freeze_bn:
        # With frozen batch norm
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),  # Frozen batch norm
            nn.ReLU(inplace=True))  # ReLU activation
    else:
        # With regular batch norm
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),  # Regular batch norm
            nn.ReLU(inplace=True))  # ReLU activation


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Modified from torchvision.misc.ops to avoid NaN issues.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))  # Weight parameter
        self.register_buffer("bias", torch.zeros(n))  # Bias parameter
        self.register_buffer("running_mean", torch.zeros(n))  # Running mean
        self.register_buffer("running_var", torch.ones(n))  # Running variance

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Custom state_dict loading that removes num_batches_tracked
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        """Forward pass with frozen normalization"""
        w = self.weight.reshape(1, -1, 1, 1)  # Reshape weight
        b = self.bias.reshape(1, -1, 1, 1)  # Reshape bias
        rv = self.running_var.reshape(1, -1, 1, 1)  # Reshape running variance
        rm = self.running_mean.reshape(1, -1, 1, 1)  # Reshape running mean
        eps = 1e-5  # Small epsilon to avoid division by zero
        scale = w * (rv + eps).rsqrt()  # Calculate normalization scale
        bias = b - rm * scale  # Calculate normalization bias
        return x * scale + bias  # Apply normalization