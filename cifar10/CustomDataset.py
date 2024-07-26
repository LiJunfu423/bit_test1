import os
import torch
from PIL import Image
import numpy as np

class CustomDataset:
    def __init__(self, root_dir,label_mapping=None):
        self.root_dir = root_dir
        self.names_list = []
        self.label_mapping = label_mapping or {}

        for dirs in os.listdir(self.root_dir):
            dir_path = os.path.join(self.root_dir, dirs)
            if dirs not in self.label_mapping:
                self.label_mapping[dirs] = len(self.label_mapping)
            for imgs in os.listdir(dir_path):
                img_path = os.path.join(dir_path, imgs)
                self.names_list.append((img_path, dirs))

    def __getitem__(self, index):
        img_path, label = self.names_list[index]
        image = Image.open(img_path)  # 打开图像
        image = image.resize((32, 32))  # 调整图像大小
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - 0.5) / 0.5  # 进一步归一化到-1到1
        # 转换为张量
        image = image.transpose((2, 0, 1))  # 重排数组维度以符合PyTorch的格式 (C, H, W)
        image = torch.tensor(image, dtype=torch.float32)  # 转换为float32类型的张量
        label_index = self.label_mapping[label]
        label_tensor = torch.tensor(label_index, dtype=torch.long)  # 转换为长整型张量
        return image, label_tensor

    def __len__(self):
        return len(self.names_list)
