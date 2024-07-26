import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from code import plot_outcome
from idCNN import myNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 模型网络
model = myNet().cuda()
# # 采用GPU训练
# model = model.to(device)  # .Cuda()数据是指放到GPU上
# model = torch.load('cifar10.pth').to(device)
# 学习率
learning_rate = 0.001
# 训练模型的次数
epoch = 40
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().cuda()
# 批处理大小
batch_size = 64
# 定义优化器
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# 准备数据集
train_data_path = './dataset/cifar-10-batches-py/train'
valid_data_path = './dataset/cifar-10-batches-py/valid'
# 数据长度
train_data_length = len(train_data_path)
valid_data_length = len(valid_data_path)

train_dataset = CustomDataset(train_data_path)
valid_dataset = CustomDataset(valid_data_path)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
train_data_size = len(train_dataset)
valid_data_size = len(valid_dataset)

# Data structure for storing training history
train_loss = []
valid_loss = []
valid_accuracy = []

total_train_step = 0
total_val_step = 0

acc_list = np.zeros(epoch)
for i in range(epoch):
    print("-----------------epoch={}-----------".format(i + 1))
    # 训练集的模型 #
    model.train()
    total_train_loss = 0
    total_train_samples = 0
    total_train_correct = 0
    for image, target in train_dataloader:
        image, target = image.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * image.size(0)
        total_train_correct += (output.argmax(1) == target).sum().item()
        total_train_samples += target.size(0)
        # print("train_times:{},Loss:{}".format(total_train_step, loss.item()))
    train_loss.append(total_train_loss / total_train_samples)
    # 验证集的模型#
    model.eval()
    total_val_loss = 0
    total_accuracy = 0
    total_val_samples = 0
    with torch.no_grad():
        for image, target in valid_dataloader:
            image, target = image.cuda(), target.cuda()
            outputs = model(image)
            loss = criterion(outputs, target)
            total_val_loss = total_val_loss + loss.item() * image.size(0)  # 计算损失值的和
            total_val_samples += target.size(0)

            accuracy = 0
            for j in target:  # 计算精确度的和
                if outputs.argmax(1)[j] == target[j]:
                    accuracy = accuracy + 1
            total_accuracy = total_accuracy + accuracy

        average_val_loss = total_val_loss / total_val_samples
        val_acc = float(total_accuracy / valid_data_size) * 100
        valid_loss.append(average_val_loss)
        valid_accuracy.append(val_acc)
        maxAcc = max(acc_list)
        acc_list[i] = val_acc
        total_val_step += 1
        # print('the_classification_is_correct :', total_accuracy, valid_data_length)
        # print("val_Loss:{}".format(total_val_loss))
        # print("val_acc:{}".format(val_acc), '%')
        # torch.save(ModelOutput.module.state_dict(), "Model_{}.pth".format(i + 1))
        if val_acc > maxAcc:
            torch.save(model, "cifar10.pth")

        print('val_max=', max(acc_list), '%', '\n')  # 验证集的最高正确率

with open('./train_loss.txt', 'w') as train_los:
    train_los.write(str(train_loss))
with open('./valid_loss.txt', 'w') as valid_los:
    valid_los.write(str(valid_loss))
with open('./valid_accuracy.txt', 'w') as valid_acc:
    valid_acc.write(str(valid_accuracy))

plot_outcome()