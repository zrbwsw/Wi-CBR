import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable 
torch_ver = torch.__version__[:3] 

# Define basic convolution module
class BasicConv(Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes

        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = torch.nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)  
        if self.bn is not None:
            x = self.bn(x)  
        if self.relu is not None:
            x = self.relu(x)  
        return x

class Flatten(Module):
    def forward(self, x):
        # Flatten input x to a one-dimensional tensor
        # x.size(0) is the current batch size, i.e., number of samples
        # -1 represents automatic dimension inference, merging all other dimensions except batch size into one dimension
        return x.view(x.size(0), -1)  # Flatten input to one dimension, prepare convolution layer output for fully connected layer

# Define channel pool module
class ChannelPool(Module):
    def forward(self, x):

        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
   
# Define spatial attention module
class SpatialGate(Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7  # Convolution kernel size
        self.compress = ChannelPool()  # Use channel pool to get channel features
        # Spatial convolution layer 1
        self.spatial1 = BasicConv(3, 3, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        # Spatial convolution layer 2, output channels is 1
        self.spatial = BasicConv(3, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=True, relu=False)

    def forward(self, x):
        x_out = self.spatial(x)  # Calculate spatial convolution
        scale = torch.sigmoid(x_out)  # Generate attention weights
        return x * scale + x, scale  # Return adjusted feature map and attention weights

# Calculate log-sum-exp 2D function
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)  # Flatten tensor
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)  # Calculate maximum value
    # Calculate log-sum-exp
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

# Define channel attention module
class ChannelGate(Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels  # Input channel number
        # Define multi-layer perceptron (MLP)
        self.mlp = torch.nn.Sequential(
            Flatten(),  # Flatten
            torch.nn.Linear(gate_channels, gate_channels // reduction_ratio),  # First linear transformation
            torch.nn.ReLU(),  # ReLU activation
            torch.nn.Linear(gate_channels // reduction_ratio, gate_channels)  # Second linear transformation
        )
        self.pool_types = pool_types  # Pool type list

    def forward(self, x):
        channel_att_sum = None  # Initialize channel attention sum
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  
                channel_att_raw = self.mlp(avg_pool)  
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  
                channel_att_raw = self.mlp(max_pool)  
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))) 
                channel_att_raw = self.mlp(lp_pool) 
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x) 
                channel_att_raw = self.mlp(lse_pool)  

            # Accumulate all channel attention
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)  # Generate channel attention weights
        return x * scale, scale  # Return adjusted feature map and attention weights
