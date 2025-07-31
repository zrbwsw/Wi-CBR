import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from tensorboardX import SummaryWriter
from load_data import BaseDataset
from load_data_cr import BaseDataset_cr
from options import Options
from da_att import SpatialGate, ChannelGate
from DPFusion import DPFusion
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ProxyContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, proxy_source='fc_weight'):
        super().__init__()
        self.temperature = temperature
        self.proxy_source = proxy_source

    def forward(self, embeddings, labels, model):
        labels = labels.long()
        if self.proxy_source == 'fc_weight':
            proxies = model.fc.weight
            
        proxies = proxies.to(embeddings.device)
        proxies = F.normalize(proxies, p=2, dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        sim_matrix = torch.matmul(embeddings, proxies.T) / self.temperature
        mask = F.one_hot(labels, num_classes=proxies.size(0)).float()
        
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        loss = -mean_log_prob.mean()
        
        return loss       

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
        self.fc = nn.Linear(1024, 6)

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

def run(device, args): 
  
    global_train_acc = []
    
    # Create trial-specific directory
    trial_dir = f'./trial_{args.trial}'
    os.makedirs(trial_dir, exist_ok=True)
    
    # Define model save directory
    model_dir = os.path.join(trial_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Define log file paths
    acc_file = os.path.join(trial_dir, 'acc.txt')
    best_acc_file = os.path.join(trial_dir, 'best_acc.txt')
    
    # TensorBoard log directory
    log_dir = os.path.join(trial_dir, 'logs')
    writer = SummaryWriter(log_dir)  
    
    phase_list = '../WIDAR_STIFMM'
    dfs_list = '../WIDAR_STIFMM_DFS'
    dataset_train = BaseDataset(
        phase_list=phase_list,
        dfs_list=dfs_list,
        split_type="indom",  
        train=True 
    )

    trainloader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker
    )
    dataset_test = BaseDataset(
        phase_list=phase_list,
        dfs_list=dfs_list,
        split_type="indom",  
        train=False 
    )

    testloader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker
    )

    net = DACN().to(device)
    ce_loss = nn.CrossEntropyLoss()
    proxy_loss = ProxyContrastiveLoss(args.temperature)

    net.to(device)
    
    params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
  
    best_acc = 89
    print("Start Training, %s!" % args.model)

    with open(acc_file, "w") as f:      
        for epoch in range(0, args.Epoch):
            print('\nEpoch: %d' % (epoch + 1))
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(trainloader, 0):
                length = len(trainloader)
                phase_inputs, dfs_inputs, labels = data
                phase_inputs, dfs_inputs, labels = phase_inputs.to(device), dfs_inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                outputs, embedding = net(phase_inputs, dfs_inputs)
                
                loss_ce = ce_loss(outputs, labels)
                loss_con = proxy_loss(embedding, labels, net)
                total_loss = loss_ce + args.beta_1 * loss_con
                
                total_loss.backward()
                optimizer.step()

                sum_loss += total_loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()

                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                        % (epoch + 1, (i + 1), sum_loss / (i + 1), 100. * correct / total))
                
                writer.add_scalar('train_loss', sum_loss / (i + 1), epoch + 1)
                global_train_acc.append(100. * correct / total)

            print("Waiting Test!")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    net.eval()                   
                    phase_inputs, dfs_inputs, labels = data
                    phase_inputs, dfs_inputs, labels = phase_inputs.to(device), dfs_inputs.to(device), labels.to(device)
                    
                    outputs, embedding = net(phase_inputs, dfs_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    
                acc = 100. * correct / total
                print('Test Acc：%.3f%%' % acc)
                
                if acc > best_acc:
                    print('Saving model weights......')
                    torch.save(net.state_dict(), os.path.join(model_dir, 'bestmodel.pth'))
                    best_acc = acc
                    best_epoch = epoch + 1
                    
                    with open(best_acc_file, "w") as f3:
                        f3.write(f"Best at Epoch {best_epoch}: {best_acc:.3f}%")
     
                f.write(f"EPOCH={epoch+1:03d}, Accuracy={acc:.3f}%\n")
                f.flush()

                writer.add_scalar('test_acc', acc, epoch + 1)
                scheduler.step()
            print(f'The best test accuracy achieved during training is: {best_acc:.3f}%')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    seed = 42
    set_seed(seed)

    args = Options().initialize()  # Initialize args here

    g = torch.Generator()
    g.manual_seed(seed)
    
    CUDA_VISIBLE_DEVICES = args.gpu_id  # Set the GPU ID to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Number of GPUs available:", torch.cuda.device_count())
    print("Using device:", device)
    
    run(device, args)