import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_res import Resnet18
from model_res_no import Resnet18_no


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_data = torchvision.datasets.CIFAR10("./dataset",
                                          train=True,
                                          download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset",
                                         train=False,
                                         download=True,
                                         transform=torchvision.transforms.ToTensor())
train_len = len(train_data)
val_len = len(test_data)


train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False, num_workers=0, drop_last=True)


# tudui = Resnet18(10)
tudui = Resnet18_no(10)


tudui = tudui.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
learning_rate = 1e-4
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)
train = 0
val = 0
epoch = 100

# writer = SummaryWriter("logs")
writer = SummaryWriter("logs_no")

best_acc = 0.0
best_epoch = 0

for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))

    tudui.train(mode=True)
    acc_ = 0
    total_train_loss = 0
    # 训练
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 数据输入模型
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化模型  清零、反向传播、优化器开始优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计训练损失
        total_train_loss += loss.item()

        # 准确率
        accuracy = (outputs.argmax(1) == targets).sum()
        acc_ += accuracy.item()

    train_loss = total_train_loss / len(train_loader)
    train_acc = acc_ / train_len
    writer.add_scalar("train_loss", train_loss, i)
    writer.add_scalar("train_acc", train_acc, i)
    print("Loss:{}, 准确率：{}".format(train_loss, train_acc))

    # 测试开关
    tudui.eval()

    # 测试
    total_test_loss = 0
    acc_val = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            # 准确率
            accuracy_val = (outputs.argmax(1) == targets).sum()
            acc_val += accuracy_val.item()

            total_test_loss += loss.item()

    test_loss = total_test_loss / len(test_loader)
    test_acc = acc_val / val_len
    writer.add_scalar("test_loss", test_loss, i)
    writer.add_scalar("test_acc", test_acc, i)
    print("整体测试集的Loss:{}, 准确率{}".format(test_loss, test_acc))

    # 保存测试准确率最高的模型
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = i + 1
        torch.save(tudui.state_dict(), "no_residuals.pth")
        print(f"新的最佳模型已保存，准确率: {best_acc:.4f}")

    print("Epoch: {} - Train Loss: {:.4f} - Train Acc: {:.4f} - Test Loss: {:.4f} - Test Acc: {:.4f}".format(
        i + 1, train_loss, train_acc, test_loss, test_acc))


    # 刷新日志
    writer.flush()
writer.close()
