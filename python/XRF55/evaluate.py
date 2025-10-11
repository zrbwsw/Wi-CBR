import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from tensorboardX import SummaryWriter
from xrf55_fine_tune_load_data_cr3 import fine_tune_cr3

from options import Options
from da_att import SpatialGate, ChannelGate
from DPFusion import DPFusion
import random
import numpy as np
import argparse

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DACN(nn.Module):
    def __init__(self):
        super(DACN, self).__init__()
        self.p_spa = SpatialGate()
        self.d_spa = SpatialGate()
        self.cga = ChannelGate(1024)

        p_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        d_resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.p_features = nn.Sequential(*list(p_resnet.children())[:-2])
        self.d_features = nn.Sequential(*list(d_resnet.children())[:-2])
        self.dp_fusion = DPFusion(1024)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, 55)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print(f'Total Parameters: {total_params:.2f}M')

    def forward(self, p_x, d_x):  
        p_outspa, p_afterspa = self.p_spa(p_x)  
        d_outspa, d_afterspa = self.d_spa(d_x) 

        p_out = self.p_features(p_outspa)
        d_out = self.d_features(d_outspa)
        
        out = torch.cat([p_out, d_out], dim=1)
        out = self.dp_fusion(out)
        
        out = self.avgpool(out)
        embedding = out.view(out.size(0), -1)
        out = self.fc(embedding)
        
        return out, embedding
    
def evaluate(device, args):

    dataset_test = fine_tune_cr3(
        phase_list=args.phase_list,
        dfs_list=args.dfs_list,
        split_type=args.split_type,  
        train=False 
    )

    testloader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker
    )

    evaluate_acc_file = os.path.join('evaluate_acc.txt')
    # 2. 初始化模型
    net = DACN().to(device)
    
    # 3. 从指定路径加载预训练的模型权重
    if os.path.exists(args.model_path):
        print(f"Loading model weights from {args.model_path}...")
        # 加载权重文件，map_location确保模型可以被正确加载到当前device
        state_dict = torch.load(args.model_path, map_location=device)
        net.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Model path not found at {args.model_path}")
        return

    # 4. 开始评估    print("Start Evaluating on the test set...")
    net.eval()  # 设置为评估模式
    
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:              
            phase_inputs, dfs_inputs, labels = data
            phase_inputs, dfs_inputs, labels = phase_inputs.to(device), dfs_inputs.to(device), labels.to(device)
            
            outputs, embedding = net(phase_inputs, dfs_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            
        acc = 100. * correct / total
        with open(evaluate_acc_file, "w") as f:
            f.write(f"Test Acc: {acc:.3f}%\n")
        print('Test Acc：%.3f%%' % acc)
        

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    parser = argparse.ArgumentParser(description='PyTorch ResNet18 示例')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/XRF55/trial_57/model/bestmodel.pth', help='Path to the saved model weights (.pth file).')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像的大小 ')
    parser.add_argument('--phase_list', type=str, default='/root/autodl-tmp/XRF55_STIFMM', help='Phase list path')
    parser.add_argument('--dfs_list', type=str, default='/root/autodl-tmp/XRF55_STIFMM_DFS', help='DFS list path')
    parser.add_argument('--split_type', type=str, default='cr4', help='Split type')
    parser.add_argument('--trial', type=int, default=668, help='试验编号')

    # 解析参数
    args = parser.parse_args()
    g = torch.Generator()
    g.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Number of GPUs available:", torch.cuda.device_count())
    print("Using device:", device)


    evaluate(device, args)