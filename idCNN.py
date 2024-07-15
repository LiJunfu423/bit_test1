import torch.nn as nn
import torch.nn.functional as F

class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.block1 =nn.Sequential (
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        # print(f"After conv3 & pool: {x.shape}")
        x = x.view(-1, 128 * 4 * 4)
        # print(f"After flattening: {x.shape}")
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # print(f"Output shape: {x.shape}")
        return x

# model = myNet()
# print(model)
# if train_on_gpu:
#     model.cuda()