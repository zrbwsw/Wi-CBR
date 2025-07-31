from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class fine_tune_cr3(Dataset):
    def __init__(self, phase_list, dfs_list, split_type="cr2", train=True):
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
      
        if self.split_type == "cr2":
            user_range = [31, 32, 33]
        elif self.split_type == "cr3":
            user_range = [34, 35, 36]
        elif self.split_type == "cr4":
            user_range = [37, 38, 39]
        else:
            raise ValueError(f"Unsupported split_type: {self.split_type}")

        if self.train:
            rn_range = [1,2]
        else:
            rn_range = range(3, 21)

        for user_id in user_range:
            for gesture_id in range(1, 56):
                for rn in rn_range:
                    filename = f"{user_id}-{gesture_id}-{rn}.jpg"
                    phase_path = os.path.join(self.phase_list, filename)
                    dfs_path = os.path.join(self.dfs_list, filename)
                    if os.path.exists(dfs_path) and os.path.exists(phase_path):
                        self.phase_img_paths.append(phase_path)
                        self.dfs_img_paths.append(dfs_path)
                        self.img_labels.append(gesture_id - 1)
                
        self.n_data = len(self.img_labels)
        if self.n_data == 0:
            raise ValueError("No images found, please check your directory structure.")

    def __getitem__(self, item):
        phase_img_path, dfs_img_path, label = self.phase_img_paths[item], self.dfs_img_paths[item], self.img_labels[item]
        phase_inputs = Image.open(phase_img_path).convert('RGB')  # 确保图像格式一致
        dfs_inputs = Image.open(dfs_img_path).convert('RGB')
        phase_inputs = self.transform(phase_inputs)
        dfs_inputs = self.transform(dfs_inputs)

        return phase_inputs, dfs_inputs, label

    def __len__(self):
        return self.n_data                   