import torch
import torch.nn.functional as F
import torch.nn as nn 

class GroupBatchnorm2d(nn.Module):
    """自定义分组批归一化层（类似GroupNorm的批标准化实现）
    
    参数:
        c_num (int): 输入特征图通道数
        group_num (int, optional): 分组数量. Defaults to 16.
        eps (float, optional): 数值稳定项. Defaults to 1e-10.
    """
    def __init__(self, c_num:int, 
                 group_num:int = 4, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num >= group_num  # 确保通道数>=分组数
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))  # 可学习缩放参数 [C,1,1]
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))    # 可学习偏置参数 [C,1,1]
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()  # 输入形状 [B,C,H,W]
        # 分组标准化计算
        x = x.view(N, self.group_num, -1)  # 重塑为 [B, G, (C/G)*H*W]
        mean = x.mean(dim=2, keepdim=True) # 计算每组均值 [B,G,1]
        std = x.std(dim=2, keepdim=True)    # 计算每组标准差 [B,G,1]
        x = (x - mean) / (std + self.eps)   # 标准化处理
        x = x.view(N, C, H, W)              # 恢复原始形状
        return x * self.weight + self.bias   # 应用缩放和平移


class DPFusion(nn.Module):
    """Spatial Reweighting Unit（空间重加权单元）
    
    参数:
        oup_channels (int): 输出通道数
        group_num (int, optional): 分组归一化的组数. Defaults to 16.
        gate_treshold (float, optional): 门控阈值. Defaults to 0.5.
        torch_gn (bool, optional): 是否使用PyTorch原生GroupNorm. Defaults to True.
    """
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 4,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True
                 ):
        super().__init__()
        # 选择归一化方式
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        # 特征重加权
        gn_x = self.gn(x)  # 分组归一化 [B,C,H,W]
        w_gamma = self.gn.weight / sum(self.gn.weight)  # 归一化权重 [C]
        w_gamma = w_gamma.view(1,-1,1,1)                # 重塑为 [1,C,1,1]
        reweigts = self.sigomid(gn_x * w_gamma)         # 空间注意力图 [B,C,H,W]
        
        # 双门控机制
        Strenth = torch.where(reweigts >= self.gate_treshold,  # 硬门控（大于阈值置1）
                        torch.ones_like(reweigts), 
                        reweigts)  # [B,C,H,W]
        Weak = torch.where(reweigts < self.gate_treshold,  # 互补门控（小于阈值置0）
                        torch.zeros_like(reweigts), 
                        reweigts)  # [B,C,H,W]

        PD_Strenth = Strenth * x  # 重要特征保留
        PD_Weak = Weak * x  # 次要特征衰减
        y = self.reconstruct(PD_Strenth, PD_Strenth)  # 特征重组
        return y

    def reconstruct(self, PD_Strenth, PD_Weak):            
        """特征交叉重组方法"""
        PS, DS = torch.split(PD_Strenth, PD_Strenth.size(1)//2, dim=1)  # 将x_1均分
        PW, DW = torch.split(PD_Weak, PD_Weak.size(1)//2, dim=1)  # 将x_2均分
        # 交叉组合：前半部分+后半部分的互补特征
        return torch.cat([PS + DS, PW + DW], dim=1)  # [B,C,H,W]

