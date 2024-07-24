import torch
from torchvision import transforms

import os
from PIL import Image

from CustomDataset import CustomDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU

model = torch.load('cifar10.pth')  # 加载模型

path = "./dataset/cifar-10-batches-py/valid/"  # 测试集

imgs = os.listdir(path)

test_num = len(imgs)


for i in imgs:
    for img_name in i:
        img = Image.open(path + i+'/'+img_name)
        test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
                                            )

        img = test_transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        outputs = model(img)  # 将图片输入到模型中
        _, predicted = outputs.max(1)

        pred_type = predicted.item()
        print(img_name, 'pred_type:', pred_type)