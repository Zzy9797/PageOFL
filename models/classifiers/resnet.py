# ResNet for CIFAR (32x32)
# 2019.07.24-Changed output of forward function
# Huawei Technologies Co., Ltd. <foss@huawei.com>
# taken from https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/DAFL/resnet.py
# for comparison with DAFL
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        out = F.relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)

        if return_features:
            return out, feature
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, channels=3):
        super(ResNet18, self).__init__()
        self.model = resnet18(num_classes=num_classes)

    def feature(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)        

        return x

    def forward(self, x, get_feature=False):
        feature = self.feature(x)
        x = self.model.fc(feature)

        if get_feature:
            return x, feature
        else:
            return x
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
        # AdaptiveAvgPool2d
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        feature = x.view(x.size(0), -1)
        x = self.fc1(x)

        if return_features:
            return x, feature
        return x

    def feat_forward(self, x):
        return self.fc1(x)

class CNN_torch(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_torch, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        feature = x.view(x.size(0), -1)
        x = self.fc3(x)

        if return_features:
            return x, feature
        return x

    def feat_forward(self, x):
        return self.fc3(x)

class CNNCifar_Five(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)  
        self.pool = nn.MaxPool2d(2, 2)    
        self.conv2 = nn.Conv2d(32, 64, 5) 
        self.fc1 = nn.Linear(64 * 5 * 5, 512)  
        self.fc2 = nn.Linear(512, num_classes) 

    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 64 * 5 * 5)          
        x = F.relu(self.fc1(x))            
        x = self.fc2(x)                   

        if return_features:
            return x, x  
        return x

class LeNet5(nn.Module):

    def __init__(self, nc=1, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, img, return_features=False):
        features = self.features( img ).view(-1, 120)
        output = self.fc( features )
        if return_features:
            return output, features
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMnist(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNMnist, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)  
        self.pool = nn.MaxPool2d(2, 2)    
    
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 5 * 5, 512)  
        self.fc2 = nn.Linear(512, num_classes) 

    def forward(self, x, return_features=False):


        

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))
        
     
        x = x.view(-1, 64 * 5 * 5)  
        

        x = F.relu(self.fc1(x))

        output = self.fc2(x)

        #print(f'Output shape: {output.shape}')
        
        if return_features:
            return output, x  
        return output

class MLP(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, input_size=32):
        super(MLP, self).__init__()


        self.flatten = nn.Flatten()

      
        self.fc1 = nn.Linear(input_channels * input_size * input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)


        self.relu = nn.ReLU()

    def forward(self, x, return_features=False):

        x = self.flatten(x)
        

        x = self.fc1(x)
        x = self.relu(x)
        

        x = self.fc2(x)
        x = self.relu(x)
        
 
        x = self.fc3(x)
        x = self.relu(x)


        x = self.fc4(x)

        if return_features:
            return x, x  
        else:
            return x  
def mlp(num_classes=10):
    return MLP(num_classes)



def cnn(num_classes=10):
    #return CNNCifar(num_classes)
    #return LeNet5(num_classes=10)
    return CNNMnist(num_classes=10)
    #return  CNNCifar_Five(num_classes)

def cnn_torch(num_classes=10):
    return CNN_torch(num_classes)

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2],num_classes )
    #return ResNet18(num_classes=10)


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)
 
def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)
 
def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)
 
def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)