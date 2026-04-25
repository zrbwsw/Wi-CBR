import torch
import torch.nn.functional as F
import torch.nn as nn 

class GroupBatchnorm2d(nn.Module):
    """Custom group batch normalization layer (batch normalization implementation similar to GroupNorm)
    
    Args:
        c_num (int): Number of input feature map channels
        group_num (int, optional): Number of groups. Defaults to 16.
        eps (float, optional): Numerical stability term. Defaults to 1e-10.
    """
    def __init__(self, c_num:int, 
                 group_num:int = 4, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num >= group_num  # Ensure number of channels >= number of groups
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))  # Learnable scaling parameter [C,1,1]
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))    # Learnable bias parameter [C,1,1]
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()  # Input shape [B,C,H,W]
        # Group normalization computation
        x = x.view(N, self.group_num, -1)  # Reshape to [B, G, (C/G)*H*W]
        mean = x.mean(dim=2, keepdim=True) # Compute mean for each group [B,G,1]
        std = x.std(dim=2, keepdim=True)    # Compute standard deviation for each group [B,G,1]
        x = (x - mean) / (std + self.eps)   # Normalization processing
        x = x.view(N, C, H, W)              # Restore original shape
        return x * self.weight + self.bias   # Apply scaling and translation


class DPFusion(nn.Module):
    """Spatial Reweighting Unit
    
    Args:
        oup_channels (int): Number of output channels
        group_num (int, optional): Number of groups for group normalization. Defaults to 16.
        gate_treshold (float, optional): Gate threshold. Defaults to 0.5.
        torch_gn (bool, optional): Whether to use PyTorch native GroupNorm. Defaults to True.
    """
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 4,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True
                 ):
        super().__init__()
        # Select normalization method
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        # Feature reweighting
        gn_x = self.gn(x)  # Group normalization [B,C,H,W]
        w_gamma = self.gn.weight / sum(self.gn.weight)  # Normalized weights [C]
        w_gamma = w_gamma.view(1,-1,1,1)                # Reshape to [1,C,1,1]
        reweigts = self.sigomid(gn_x * w_gamma)         # Spatial attention map [B,C,H,W]
        
        # Dual gating mechanism
        Strenth = torch.where(reweigts >= self.gate_treshold,  # Hard gating (set to 1 if above threshold)
                        torch.ones_like(reweigts), 
                        reweigts)  # [B,C,H,W]
        Weak = torch.where(reweigts < self.gate_treshold,  # Complementary gating (set to 0 if below threshold)
                        torch.zeros_like(reweigts), 
                        reweigts)  # [B,C,H,W]

        PD_Strenth = Strenth * x  # Important feature preservation
        PD_Weak = Weak * x  # Secondary feature attenuation
        y = self.reconstruct(PD_Strenth, PD_Weak)  # Feature reconstruction
        return y

    def reconstruct(self, PD_Strenth, PD_Weak):            
        """Feature cross-reconstruction method"""
        PS, DS = torch.split(PD_Strenth, PD_Strenth.size(1)//2, dim=1)  # Split x_1 into two halves
        PW, DW = torch.split(PD_Weak, PD_Weak.size(1)//2, dim=1)  # Split x_2 into two halves
        # Cross combination: first half + complementary features of second half
        return torch.cat([PS + DW, PW + DS], dim=1)  # [B,C,H,W]

