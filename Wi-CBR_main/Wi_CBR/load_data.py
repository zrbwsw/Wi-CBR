from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class BaseDataset(Dataset):
    def __init__(self, phase_list, dfs_list, split_type="indom", train=True):
        self.phase_list = phase_list
        self.dfs_list = dfs_list
        self.split_type = split_type
        self.train = train
        self.phase_img_paths = []
        self.dfs_img_paths = []
        self.img_labels = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self._load_data()

    def _load_data(self):
        if self.split_type == "indom":
            rn_range = [1,2,3,4] if self.train else [5]
            ln_range, on_range = range(1, 6), range(1, 6)
        elif self.split_type == "cl":
            ln_range = [1,2,3,4] if self.train else [5]
            rn_range, on_range = range(1, 6), range(1, 6)
        elif self.split_type == "co":
            on_range = [1,2,3,4] if self.train else [5]
            rn_range, ln_range = range(1, 6), range(1, 6)
        else:
            raise ValueError(f"Unsupported split_type: {self.split_type}")

        # 遍历加载文件
        for user_id in range(0, 9):  # 使用请求的用户数量
            for gesture_id in range(1, 7):  # 使用请求的手势数量  
                for ln in ln_range:  # 位置编号范围
                    for on in on_range:  # 朝向编号范围
                        for rn in rn_range:  # 重复次数
                            filename = f"{user_id}-{gesture_id}-{ln}-{on}-{rn}.jpg"          
                            phase_path = os.path.join(self.phase_list, filename)
                            dfs_path = os.path.join(self.dfs_list, filename)

                            self.phase_img_paths.append(phase_path)
                            self.dfs_img_paths.append(dfs_path)
                            self.img_labels.append(gesture_id - 1)

        self.n_data = len(self.img_labels)

    def __getitem__(self, item):
        phase_img_path, dfs_img_path, label = self.phase_img_paths[item], self.dfs_img_paths[item], self.img_labels[item]
        phase_inputs = Image.open(phase_img_path).convert('RGB')  # 确保图像格式一致
        dfs_inputs = Image.open(dfs_img_path).convert('RGB')
        phase_inputs = self.transform(phase_inputs)
        dfs_inputs = self.transform(dfs_inputs)

        return phase_inputs, dfs_inputs, label

    def __len__(self):
        return self.n_data