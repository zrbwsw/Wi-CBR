import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable #从 PyTorch 的自动求导模块中导入 Variable 类
torch_ver = torch.__version__[:3]# 切片操作， 获取当前torch版本号

# 定义基本卷积模块
class BasicConv(Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # 卷积层
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # 批归一化层（可选）
        self.bn = torch.nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        # ReLU激活函数（可选）
        self.relu = torch.nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)  # 进行卷积操作，将输入 x 传递给卷积层 self.conv进行卷积操作
        if self.bn is not None:
            x = self.bn(x)  # 如果 self.bn 不是 None，说明我们要应用批归一化
        if self.relu is not None:
            x = self.relu(x)  # 应用ReLU激活
        return x

# 定义展平模块
class Flatten(Module):
    def forward(self, x):
        # 将输入 x 展平为一维张量
        # x.size(0) 是当前批次的大小，即样本数量
        # -1 代表自动推断维度，这样可以将所有其他维度，除了批次大小，合并为一维
        return x.view(x.size(0), -1)  # 将输入展平为一维，将卷积层的输出准备好以输入到全连接层

# 定义通道池模块
class ChannelPool(Module):
    def forward(self, x):
        # 在通道维度上进行最大池化和平均池化，然后合并结果
        # torch.max(x, 1)[0] 返回每个通道的最大值，unsqueeze(1) 增加维度以便合并
        #用于在指定位置（这里是通道维度）添加一个维度，使得结果保持原来的维度，同时在该位置的大小为 1。这通常用于保持张量的形状一致性，便于后续操作，如拼接或广播
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # 在 torch.max 的返回值中，第一个元素（索引为 0）是最大值，第二个元素（索引为 1）是最大值的索引

# 定义空间注意力模块
class SpatialGate(Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7  # 卷积核大小
        self.compress = ChannelPool()  # 使用通道池来获取通道特征
        # 空间卷积层1
        self.spatial1 = BasicConv(3, 3, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        # 空间卷积层2，输出通道为1
        self.spatial = BasicConv(3, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=True, relu=False)

    def forward(self, x):
        x_out = self.spatial(x)  # 计算空间卷积
        scale = torch.sigmoid(x_out)  # 生成注意力权重
        return x * scale + x, scale  # 返回调整后的特征图和注意力权重

# 计算log-sum-exp的二维函数
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)  # 将张量展平
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)  # 计算最大值
    # 计算log-sum-exp
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

# 定义通道注意力模块
class ChannelGate(Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels  # 输入通道数
        # 定义多层感知机（MLP）
        self.mlp = torch.nn.Sequential(
            Flatten(),  # 展平
            torch.nn.Linear(gate_channels, gate_channels // reduction_ratio),  # 第一层线性变换
            torch.nn.ReLU(),  # ReLU激活
            torch.nn.Linear(gate_channels // reduction_ratio, gate_channels)  # 第二层线性变换
        )
        self.pool_types = pool_types  # 池化类型列表

    def forward(self, x):
        channel_att_sum = None  # 初始化通道注意力和
        for pool_type in self.pool_types:
            # 根据池化类型计算通道注意力
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 全局平均池化
                channel_att_raw = self.mlp(avg_pool)  # MLP处理
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # 全局最大池化
                channel_att_raw = self.mlp(max_pool)  # MLP处理
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # Lp池化
                channel_att_raw = self.mlp(lp_pool)  # MLP处理
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)  # log-sum-exp池化
                channel_att_raw = self.mlp(lse_pool)  # MLP处理

            # 累加所有通道注意力
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)  # 生成通道注意力权重
        return x * scale, scale  # 返回调整后的特征图和注意力权重
