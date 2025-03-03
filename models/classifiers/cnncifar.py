from torch import nn
import torch.nn.functional as F

class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        feature = x.view(x.size(0), -1)
        x = self.fc1(x)
        if return_features:
            return x, feature
        return x

    def feat_forward(self, x):
        return self.fc1(x)


class CNNMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNMNIST, self).__init__()
        # 将输入通道数从3改为1，因为MNIST是灰度图像
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        # 经过两次池化操作，特征图的大小变为7x7，计算fc1的输入维度时需要考虑这一点
        self.fc1 = nn.Linear(128 * 7 * 7, num_classes)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> Pooling
        x = F.relu(self.conv3(x))             # Conv3 (不加池化层)
        x = x.view(-1, 128 * 7 * 7)           # Flattening before passing to FC layer
        feature = x.view(x.size(0), -1)
        x = self.fc1(x)                       # Fully connected layer
        if return_features:
            return x, feature
        return x

    def feat_forward(self, x):
        return self.fc1(x)



def cnn(num_classes=10):
    return CNNMNIST(num_classes)