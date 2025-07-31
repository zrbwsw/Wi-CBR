import argparse

class Options():
    def initialize(self):
        parser = argparse.ArgumentParser(description='PyTorch ResNet18 示例')

        # 训练相关超参数
        parser.add_argument('--lr', type=float, default=0.0001, help='SGD 优化器的初始学习率')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD 的动量')
        parser.add_argument('--train_batch_size', type=int, default=10, metavar='N', help='训练时的批量大小 (默认: 128)')
        parser.add_argument('--test_batch_size', type=int, default=10, metavar='N', help='测试时的批量大小 (默认: 128)')
        parser.add_argument('--Epoch', type=int, default=30, help='训练的总轮数 (默认: 200)')
        parser.add_argument('--model', type=str, default='resnet18', help='要训练的模型名称 (默认: resnet18)')
        parser.add_argument('--img_size', type=int, default=224, help='输入图像的大小 (默认: 224)')
        parser.add_argument('--temperature', type=float, default=0.1, help='ProxyContrastiveLoss 中的温度系数 (默认: 0.1)')
        parser.add_argument('--beta_1', type=float, default=0.1, help='对比损失在总损失中的权重 (默认: 0.1)')
        parser.add_argument('--trial', type=int, default=57, help='试验编号')
        # 解析参数
        args = parser.parse_args()

        return args