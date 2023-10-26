import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import Utils

# 本类用于训练使用
# by wanbiao
## 请注意，本类，如果要使用在mac上，需要将device替换成对应的mps
class PlantTrain:
    def __init__(self,net,trainData,learnRate=0.01,momentum=0.9,weightDecay=5e-4,epoch=1) -> None:
        self.net = net
        self.trainData = trainData
        self.learnRate = learnRate
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.epoch = epoch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.createLoss()
        self.createOptimizer()

    def createLoss(self):
        self.loss = nn.CrossEntropyLoss()
        self.loss.to(self.device)

    def createOptimizer(self):
        self.optimizer = optim.SGD(self.net.parameters(), self.learnRate,self.momentum, self.weightDecay)

    def startTrain(self,saveMode=False):
        # 训练模型
        for epoch in  tqdm(range(self.epoch)):
            for i, data in tqdm(enumerate(self.trainData, 0)):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 清零梯度缓存
                self.optimizer.zero_grad()

                # 前向传播，计算损失，反向传播
                outputs = self.net(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()

                # 更新参数
                self.optimizer.step()
            # 切换模型为评估模式
            self.net.eval()
            self.eval()
            # 切换会训练模式
            self.net.train()
            if saveMode:
                modelName = f'epoch_{epoch}.pt'
                Utils.PlantUtil().saveModel(self.net,modelName)
        print("train successfully\n")
    
    def eval(self):
        correct = 0
        total = 0
        # 使用了整个训练集进行评估
        with torch.no_grad():
            for i,data in enumerate(self.trainData, 0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                # 取出最大值
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')
