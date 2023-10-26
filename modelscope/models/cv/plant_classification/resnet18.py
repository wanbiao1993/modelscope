import torch
import torch.nn as nn


# 本文件定义ResNet相关结构
# by wanbiao
class ResidualBlock(nn.Module):
    # inChannels表示输入通道数
    # outChannels表示输出通道数
    # stride卷积步幅
    def __init__(self, inChannels, outChannels, stride=1):
        super(ResidualBlock, self).__init__()
        # 第一个卷积
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1)
        # 使用batchnormal代替池化
        self.bn1 = nn.BatchNorm2d(outChannels)
        # 激活函数为relu
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积，他的输入为第一的输出
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1)
        # 依然用batchnormal代替池化
        self.bn2 = nn.BatchNorm2d(outChannels)
        
        # 构建shortcut
        self.shortcut = nn.Sequential()
        ## 维度不同时，1x1卷积进行升维
        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outChannels)
            )
    # 前向传播时被调用，x为输入值        
    def forward(self, x):
        # 先是第一层卷积，再是batchnormal再是relu
        out = self.relu(self.bn1(self.conv1(x)))
        # 接着第二层卷积，再是batchnormal
        out = self.bn2(self.conv2(out))
        # 输入x直接经过shorcut与out相加
        out += self.shortcut(x)
        # 最后一个relu输出
        out = self.relu(out)
        return out


# 构建ResNet18对象
class ResNet18(nn.Module):

    # num
    def __init__(self, numClasses=0):
        super(ResNet18, self).__init__()
        # 因为为RGB所以这里是3通道，输出固定为64通道
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # 构建batchnormal对象 
        self.bn1 = nn.BatchNorm2d(64)
        # 构建relu对象
        self.relu = nn.ReLU(inplace=True)
        # resnet中的第一层，输入为64通道，输出为64通道
        # 下面的layer2，layer3，layer4一样，只是通道不同
        self.layer1 = self.makeLayer(64, 64, 1)
        self.layer2 = self.makeLayer(64, 128, 1, stride=2)
        self.layer3 = self.makeLayer(128, 256, 1, stride=2)
        self.layer4 = self.makeLayer(256, 512, 1, stride=2)
        # 创建自适应平均池化输出为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 一个全连接层,输出为类的个数
        self.fc = nn.Linear(512, numClasses)

    # 前向传播时调用
    def forward(self, x):
        # 先是第一个卷积层
        x = self.conv1(x)
        # 然后是batchnormal
        x = self.bn1(x)
        # 接着relu
        x = self.relu(x)
        # layer1内部有一个残差块
        x = self.layer1(x)
        # layer2内部也有一个残差快
        x = self.layer2(x)
        # layer3内部依然有一个残差块
        x = self.layer3(x)
        # layer4依然有一个残差块
        x = self.layer4(x)
        # 自适应平均池化
        x = self.avgpool(x)
        # 将张量展开       
        x = torch.flatten(x, 1)
        # 输入给全连接层
        x = self.fc(x)
        # 注意此处没有进行softmax,因此最后的结果需要自己softMax
        return x

    # 创建残差块，分别传递输入，输出通道，以及残差块个数
    def makeLayer(self, inChannels, outChannels, blocks, stride=1):
        layers = []
        # 创建残差快对象
        layers.append(ResidualBlock(inChannels, outChannels, stride))
        for _ in range(0, blocks):
            layers.append(ResidualBlock(outChannels, outChannels))
        return nn.Sequential(*layers)
