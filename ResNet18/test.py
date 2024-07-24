import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.utils import data
from model_res import Resnet18
from model_res_no import Resnet18_no

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = Resnet18(10)
net = Resnet18_no(10)

load_path = "no_residuals.pth"
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)
batch_size = 128
transform = transforms.Compose([transforms.ToTensor()])
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
test_dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=transform)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
data_loaders = {
    "test": test_dataloader
}
net.to(device)
net.eval()
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in tqdm(data_loaders['test']):
    imgs, labels = data
    imgs = imgs.to(device)
    labels = labels.to(device)
    outputs = net(imgs)
    _, preds = torch.max(outputs, 1)
    c = (preds == labels)
    c = c.squeeze()
    for i in range(len(labels)):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(10)]
overall_accuracy = 100 * sum(class_correct) / sum(class_total)

for i in range(10):
    print(
        f"Accuracy of {classes[i]:>10} : {np.round(100 * class_correct[i].detach().cpu().numpy() / class_total[i], 2)}%")

plt.figure(figsize=(12, 6))
plt.bar(classes, class_accuracies, color='skyblue')
plt.axhline(y=overall_accuracy, color='r', linestyle='--', label=f'Overall accuracy: {overall_accuracy:.2f}%')
plt.xlabel('Classes')
plt.ylabel('Accuracy (%)')
# plt.title('The Accuracy of ResNet18 with residuals ON CIFAR-10')
plt.title('The Accuracy of ResNet18 without residuals ON CIFAR-10')
plt.legend()
plt.show()