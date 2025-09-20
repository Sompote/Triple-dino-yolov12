# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "TripleInputConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input."""

    def __init__(self, c1, c2, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        return x[self.index]


class TripleInputConv(nn.Module):
    """Triple input convolution layer that processes 9-channel input as three separate 3-channel images.
    
    Supports loading pretrained weights from standard YOLOv12 models by replicating the first layer weights
    across the three branches and fine-tuning from there.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, pretrained_weights=None):
        """
        Initialize TripleInputConv layer.
        
        Args:
            c1: Input channels (should be 9 for triple input)
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups (will be set to 1 for the branch convolutions)
            d: Dilation
            act: Activation function
            pretrained_weights: Optional pretrained weights from standard YOLOv12 first layer
        """
        super().__init__()
        # Ensure we have 9 input channels for triple input
        assert c1 == 9, f"TripleInputConv expects 9 input channels, got {c1}"
        
        # Store initialization parameters
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.d = d
        self.act_type = act
        
        # Three separate convolution branches for each 3-channel input
        # Use g=1 for branch convolutions to avoid group compatibility issues
        self.conv1 = Conv(3, c2, k, s, p, 1, d, act)
        self.conv2 = Conv(3, c2, k, s, p, 1, d, act)
        self.conv3 = Conv(3, c2, k, s, p, 1, d, act)
        
        # Optional fusion layer to combine features from three branches
        self.fusion = Conv(c2 * 3, c2, 1, 1, 0, 1, 1, act)
        
        # Initialize weights from pretrained model if provided
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def forward(self, x):
        """
        Forward pass for triple input.
        
        Args:
            x: Input tensor with shape (batch, 9, height, width)
            
        Returns:
            Fused features from three input branches
        """
        # Split 9-channel input into three 3-channel branches
        x1 = x[:, 0:3, :, :]  # First image (channels 0-2)
        x2 = x[:, 3:6, :, :]  # Second image (channels 3-5)  
        x3 = x[:, 6:9, :, :]  # Third image (channels 6-8)
        
        # Process each branch independently
        out1 = self.conv1(x1)
        out2 = self.conv2(x2)
        out3 = self.conv3(x3)
        
        # Concatenate and fuse features
        combined = torch.cat([out1, out2, out3], dim=1)
        fused = self.fusion(combined)
        
        return fused
    
    def load_pretrained_weights(self, pretrained_state_dict):
        """
        Load pretrained weights from a standard YOLOv12 model.
        
        Args:
            pretrained_state_dict: State dict from pretrained YOLOv12 model
        """
        try:
            # Extract first layer weights from pretrained model
            # Look for the first conv layer (typically 'model.0.conv.weight')
            first_layer_key = None
            for key in pretrained_state_dict.keys():
                if 'model.0.' in key and 'conv.weight' in key:
                    first_layer_key = key
                    break
            
            if first_layer_key is None:
                print("Warning: Could not find first layer weights in pretrained model")
                return
            
            pretrained_weight = pretrained_state_dict[first_layer_key]
            pretrained_bias_key = first_layer_key.replace('weight', 'bias')
            pretrained_bias = pretrained_state_dict.get(pretrained_bias_key)
            
            # Load weights into all three branches
            self._load_branch_weights(self.conv1, pretrained_weight, pretrained_bias, pretrained_state_dict, first_layer_key)
            self._load_branch_weights(self.conv2, pretrained_weight, pretrained_bias, pretrained_state_dict, first_layer_key)
            self._load_branch_weights(self.conv3, pretrained_weight, pretrained_bias, pretrained_state_dict, first_layer_key)
            
            print(f"✓ Loaded pretrained weights from {first_layer_key} into TripleInputConv branches")
            
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
    
    def _load_branch_weights(self, branch_conv, pretrained_weight, pretrained_bias, full_state_dict, base_key):
        """Load pretrained weights into a single branch."""
        try:
            # Load conv weights
            if pretrained_weight.shape[1] == 3:  # Standard 3-channel input
                with torch.no_grad():
                    branch_conv.conv.weight.copy_(pretrained_weight)
                    if pretrained_bias is not None:
                        branch_conv.conv.bias.copy_(pretrained_bias)
            
            # Load BatchNorm weights if available
            bn_weight_key = base_key.replace('conv.weight', 'bn.weight')
            bn_bias_key = base_key.replace('conv.weight', 'bn.bias')
            bn_mean_key = base_key.replace('conv.weight', 'bn.running_mean')
            bn_var_key = base_key.replace('conv.weight', 'bn.running_var')
            
            if bn_weight_key in full_state_dict:
                with torch.no_grad():
                    branch_conv.bn.weight.copy_(full_state_dict[bn_weight_key])
                    branch_conv.bn.bias.copy_(full_state_dict[bn_bias_key])
                    branch_conv.bn.running_mean.copy_(full_state_dict[bn_mean_key])
                    branch_conv.bn.running_var.copy_(full_state_dict[bn_var_key])
                    
        except Exception as e:
            print(f"Warning: Failed to load weights for branch: {e}")
    
    @classmethod
    def from_pretrained(cls, c1, c2, pretrained_model_path, **kwargs):
        """
        Create TripleInputConv layer with pretrained weights.
        
        Args:
            c1: Input channels (should be 9)
            c2: Output channels
            pretrained_model_path: Path to pretrained YOLOv12 model (.pt file)
            **kwargs: Additional arguments for TripleInputConv
            
        Returns:
            TripleInputConv layer with loaded pretrained weights
        """
        import torch
        
        # Load pretrained model
        try:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Create layer with pretrained weights
            layer = cls(c1, c2, pretrained_weights=state_dict, **kwargs)
            return layer
            
        except Exception as e:
            print(f"Warning: Failed to load pretrained model from {pretrained_model_path}: {e}")
            # Return layer without pretrained weights
            return cls(c1, c2, **kwargs)
