import numpy as np
import torch
from PIL import Image

model = torch.load('cifar10.pth', map_location=torch.device('cpu'))
def classify_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = Image.fromarray(image).resize((32, 32))  # 调整图像大小
    image = np.array(image, dtype=np.float32) / 255.0
    image = (image - 0.5) / 0.5  # 进一步归一化到-1到1
    image = image.transpose((2, 0, 1))  # 重排数组维度以符合PyTorch的格式 (C, H, W)
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)

    with torch.no_grad():
        prediction = model(image)
        predicted_index = prediction.argmax(1).item()
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        result = class_names[predicted_index]
        return result
