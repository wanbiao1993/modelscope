import torchvision.transforms as transforms
from modelscope.msdatasets import MsDataset
import torch
import DataSet
import ResNet18
import Train


# 训练的主入口


## 加载训练数据
batchSize = 10
msDataTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
msDataTrain = MsDataset.load('AiguLiu/plants', subset_name='default', split='train')
msDataTrainLoader = torch.utils.data.DataLoader(msDataTrain.to_torch_dataset(), batch_size=batchSize, shuffle=False)
dataTrainLoader = torch.utils.data.DataLoader(DataSet.CustomDataset(msDataTrainLoader,transform=msDataTransform),batch_size=batchSize, shuffle=False)

## 加载测试数据
msDataTest = MsDataset.load('AiguLiu/plants', subset_name='default', split='validation')
msDataTestLoader = torch.utils.data.DataLoader(msDataTest.to_torch_dataset(), batch_size=batchSize, shuffle=False)
dataTestLoader = torch.utils.data.DataLoader(DataSet.CustomDataset(msDataTestLoader,transform=msDataTransform),batch_size=batchSize, shuffle=False)

## 构建网络并训练
model=ResNet18.ResNet18(30)
train=Train.PlantTrain(model,dataTrainLoader,epoch=50)
train.startTrain(True)
