import argparse

class Options():
    def initialize(self):
        parser = argparse.ArgumentParser(description='PyTorch ResNet18 Training Framework')

        # Training hyperparameters
        parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
        parser.add_argument('--train_batch_size', type=int, default=10, metavar='N', help='Training batch size')
        parser.add_argument('--test_batch_size', type=int, default=10, metavar='N', help='Testing batch size')
        parser.add_argument('--Epoch', type=int, default=30, help='Total number of training epochs')  
        parser.add_argument('--model', type=str, default='resnet18', help='Model architecture to train')
        parser.add_argument('--img_size', type=int, default=224, help='Input image size')
        parser.add_argument('--temperature', type=float, default=0.1, help='Temperature parameter for ProxyContrastiveLoss')
        parser.add_argument('--beta_1', type=float, default=0.1, help='Weight coefficient for contrastive loss in total loss')
        parser.add_argument('--gpu_id', type=str, default='0', help='GPU device ID to use')
        parser.add_argument('--trial', type=int, default=1, help='Trial number for experiment tracking')
        # Parse arguments
        args = parser.parse_args()

        return args