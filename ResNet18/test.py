import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model_res import Resnet18
from model_res_no import Resnet18_no


def main():
    # 使用 GPU 如果可用的话
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 测试集数据归一化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载测试数据集
    test_data = torchvision.datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transform_test)

    # 数据加载器
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=128, shuffle=False, num_workers=2)

    # 加载模型
    model = Resnet18_no(10)  # 确保使用与你训练时相同的模型结构
    model = model.to(device)
    model.eval()

    # 加载模型权重
    model.load_state_dict(torch.load("no_residuals.pth"))

    # 初始化列表用于记录每个类别的正确和总预测数
    class_correct = [0.0 for _ in range(10)]
    class_total = [0.0 for _ in range(10)]
    total_correct = 0  # 总正确预测数
    total_samples = 0  # 总样本数

    # 评估模型
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            # 跟踪每个类别的正确预测数和总样本数
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1
                total_correct += (predicted[i] == targets[i]).item()
                total_samples += 1

    # 计算每个类别的准确率
    class_accuracy = [100 * (correct / total) if total > 0 else 0 for correct, total in zip(class_correct, class_total)]
    overall_accuracy = 100 * total_correct / total_samples

    # 打印每个类别的准确率
    classes = test_data.classes
    for i, accuracy in enumerate(class_accuracy):
        print(f"Class {classes[i]} Accuracy: {accuracy:.2f}%")

    # 打印整体准确率
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # 绘制柱状图
    plt.figure(figsize=(10, 5))
    plt.bar(classes, class_accuracy, color='skyblue')
    plt.ylabel('%')
    plt.title(f'The Accuracy of ResNet18 without residuals ON CIFAR-10\nOverall Accuracy: {overall_accuracy:.2f}%')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == '__main__':
    main()
