import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from model_res18 import ResNet18
from model_res18_no import ResNet18_no

log_dir = "logs_res18_no"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = torchvision.datasets.CIFAR10("./dataset",
                                          train=True,
                                          download=True,
                                          transform=transform_train)
test_data = torchvision.datasets.CIFAR10("./dataset",
                                         train=False,
                                         download=True,
                                         transform=transform_test)
train_len = len(train_data)
val_len = len(test_data)

train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=False, num_workers=0, drop_last=True)

# tudui = ResNet18()
tudui = ResNet18_no()

tudui = tudui.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
epoch = 140

writer = SummaryWriter(log_dir)
best_acc = 0.0
best_epoch = 0

for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))

    tudui.train()
    acc_ = 0
    total_train_loss = 0

    # 包装训练数据加载器
    train_loader_tqdm = tqdm(train_loader, desc=f"训练轮次 {i + 1}/{epoch}", unit="batch")

    # 训练
    for data in train_loader_tqdm:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        accuracy = (outputs.argmax(1) == targets).sum()
        acc_ += accuracy.item()

        # 更新进度条
        train_loader_tqdm.set_postfix(loss=loss.item(), acc=accuracy.item() / len(targets))

    train_loss = total_train_loss / len(train_loader)
    train_acc = acc_ / train_len
    writer.add_scalar("train_loss", train_loss, i)
    writer.add_scalar("train_acc", train_acc, i)
    print("Loss:{}, 准确率：{}".format(train_loss, train_acc))

    tudui.eval()
    total_test_loss = 0
    acc_val = 0

    # 包装测试数据加载器
    test_loader_tqdm = tqdm(test_loader, desc=f"测试轮次 {i + 1}/{epoch}", unit="batch")

    # 测试
    with torch.no_grad():
        for data in test_loader_tqdm:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            accuracy_val = (outputs.argmax(1) == targets).sum()
            acc_val += accuracy_val.item()
            total_test_loss += loss.item()

            # 更新进度条
            test_loader_tqdm.set_postfix(loss=loss.item(), acc=accuracy_val.item() / len(targets))

    test_loss = total_test_loss / len(test_loader)
    test_acc = acc_val / val_len
    writer.add_scalar("test_loss", test_loss, i)
    writer.add_scalar("test_acc", test_acc, i)
    print("整体测试集的Loss:{}, 准确率{}".format(test_loss, test_acc))

    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = i + 1
        torch.save(tudui.state_dict(), "no_residuals.pth")
        print(f"新的最佳模型已保存，准确率: {best_acc:.4f}")

    print("Epoch: {} - Train Loss: {:.4f} - Train Acc: {:.4f} - Test Loss: {:.4f} - Test Acc: {:.4f}".format(
        i + 1, train_loss, train_acc, test_loss, test_acc))

    writer.flush()
writer.close()
