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
from model_profiler import ModelProfiler
from f1_calculator import F1Calculator
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
    
    phase_list = '../../WIDAR_STIFMM'
    dfs_list = '../../WIDAR_STIFMM_DFS'
    dataset_train = BaseDataset_cr(
        phase_list=phase_list,
        dfs_list=dfs_list,
        split_type="cr3",  
        train=True 
    )

    trainloader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker
    )
    dataset_test = BaseDataset_cr(
        phase_list=phase_list,
        dfs_list=dfs_list,
        split_type="cr3",  
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
    
    # Model profiling - measure FLOPs and inference time
    print("\n" + "="*60)
    print("PERFORMING MODEL PROFILING")
    print("="*60)
    
    profiler = ModelProfiler(net, device)
    # Profile with typical input size (batch_size=1 for single inference)
    profile_results = profiler.profile_model(
        input_shape=(1, 3, 224, 224),  # Single sample profiling
        num_classes=6,
        num_runs=100
    )
    profiler.print_profile_results(profile_results)
    
    # Save profiling results to file
    profile_file = os.path.join(trial_dir, 'model_profile.txt')
    with open(profile_file, 'w') as f:
        f.write("MODEL PROFILING RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"FLOPs: {profile_results['flops_g']:.2f} G\n")
        f.write(f"Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.2f} M\n")
        
        timing = profile_results['inference_time']
        f.write(f"\nInference Time Statistics:\n")
        f.write(f"  Mean: {timing['mean_ms']:.2f} ms\n")
        f.write(f"  Std:  {timing['std_ms']:.2f} ms\n")
        f.write(f"  Min:  {timing['min_ms']:.2f} ms\n")
        f.write(f"  Max:  {timing['max_ms']:.2f} ms\n")
        f.write(f"  Median: {timing['median_ms']:.2f} ms\n")
        
        info = profile_results['model_info']
        f.write(f"\nProfiling Configuration:\n")
        f.write(f"  Input Shape: {info['input_shape']}\n")
        f.write(f"  Device: {info['device']}\n")
        f.write(f"  Timing Runs: {info['num_timing_runs']}\n")
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
  
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    print("Start Training, %s!" % args.model)

    with open(acc_file, "w") as f:      
        for epoch in range(0, args.Epoch):
            print('\nEpoch: %d' % (epoch + 1))
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(trainloader, 0):

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
                
                writer.add_scalar('train_loss', sum_loss / (i + 1), epoch + 1)
                global_train_acc.append(100. * correct / total)

            # Print training results once per epoch
            train_acc = 100. * correct / total
            avg_loss = sum_loss / len(trainloader)
            print('Training - Loss: %.03f | Acc: %.3f%%' % (avg_loss, train_acc))

            print("Starting evaluation...")
            with torch.no_grad():
                correct = 0
                total = 0
                
                # Initialize F1 calculator for test evaluation
                f1_calculator = F1Calculator(num_classes=6, class_names=[
                    'Push-Pull', 'Sweep', 'Clap', 'Slide', 'Draw-O(Horizontal)', 'Draw-Zigzag(Horizontal)'
                ])
                
                for data in testloader:
                    net.eval()                   
                    phase_inputs, dfs_inputs, labels = data
                    phase_inputs, dfs_inputs, labels = phase_inputs.to(device), dfs_inputs.to(device), labels.to(device)
                    
                    outputs, embedding = net(phase_inputs, dfs_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    
                    # Update F1 calculator
                    f1_calculator.update(outputs, labels)
                    
                acc = 100. * correct / total
                
                # Calculate F1 scores
                f1_metrics = f1_calculator.compute_f1_scores()
                macro_f1 = f1_metrics['macro_f1']
                
                print('Test Accuracy: %.3f%% | Macro F1: %.4f' % (acc, macro_f1))
                
                # Update best metrics tracking
                if acc > best_acc:
                    print('Saving best model weights...')
                    torch.save(net.state_dict(), os.path.join(model_dir, 'bestmodel.pth'))
                    best_acc = acc
                    best_f1 = macro_f1
                    best_epoch = epoch + 1
                    
                    with open(best_acc_file, "w") as f3:
                        f3.write(f"Best at Epoch {best_epoch}: Accuracy={best_acc:.3f}%, Macro F1={best_f1:.4f}")
                    
                    # Save detailed F1 results for best model
                    f1_results_file = os.path.join(trial_dir, 'best_f1_results.txt')
                    f1_calculator.save_results_to_file(f1_results_file)
     
                f.write(f"EPOCH={epoch+1:03d}, Accuracy={acc:.3f}%, Macro_F1={macro_f1:.4f}\n")
                f.flush()

                writer.add_scalar('test_acc', acc, epoch + 1)
                writer.add_scalar('test_macro_f1', macro_f1, epoch + 1)
                scheduler.step()
        
        # Print final best results
        print("\n" + "="*60)
        print("TRAINING COMPLETED - FINAL RESULTS")
        print("="*60)
        print(f'Best Test Accuracy: {best_acc:.3f}% (Epoch {best_epoch})')
        print(f'Best Macro F1 Score: {best_f1:.4f} (Epoch {best_epoch})')
        print("="*60)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    seed = 888
    set_seed(seed)

    args = Options().initialize()  # Initialize args here

    g = torch.Generator()
    g.manual_seed(seed)
    
    # Set the GPU 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run(device, args)