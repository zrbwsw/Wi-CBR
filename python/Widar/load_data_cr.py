from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class BaseDataset_cr(Dataset):
    def __init__(self, phase_list, dfs_list, split_type="cr1", train=True):
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
        if self.split_type == "cr1":
            cerange = [9, 10, 11, 12, 13, 14, 15, 16] if self.train else [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif self.split_type == "cr2":
            cerange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14] if self.train else [9, 10, 15, 16] 
        elif self.split_type == "cr3":
            cerange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16] if self.train else [11, 12, 13, 14]        
        else:
            raise ValueError(f"Unsupported split_type: {self.split_type}")

        # 遍历加载文件
        for user_id in cerange:  # 使用请求的用户数量
            for gesture_id in range(1, 7):  # 使用请求的手势数量
                for ln in range(1, 6):  # 位置编号范围
                    for on in range(1, 6):  # 朝向编号范围
                        for rn in range(1, 6):  # 重复次数
                            filename = f"{user_id}-{gesture_id}-{ln}-{on}-{rn}.jpg"          
                            phase_path = os.path.join(self.phase_list, filename)
                            dfs_path = os.path.join(self.dfs_list, filename)          
                            if os.path.exists(dfs_path):
                                self.phase_img_paths.append(phase_path)
                                self.dfs_img_paths.append(dfs_path)
                                self.img_labels.append(gesture_id - 1)

        self.n_data = len(self.img_labels)
        if self.n_data == 0:
            raise ValueError("No images found, please check your directory structure.")

    def __getitem__(self, item):
        phase_img_path, dfs_img_path, label = self.phase_img_paths[item], self.dfs_img_paths[item], self.img_labels[item]
        try:
            phase_inputs = Image.open(phase_img_path).convert('RGB')
            dfs_inputs = Image.open(dfs_img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error while opening image files {phase_img_path} or {dfs_img_path}: {e}")

        phase_inputs = self.transform(phase_inputs)
        dfs_inputs = self.transform(dfs_inputs)

        return phase_inputs, dfs_inputs, label

    def __len__(self):
        return self.n_data                   