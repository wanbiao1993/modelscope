from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# 自定义数据集，该数据集，接收ModelScope中的DataLoader,并加载图片进内存
# by wanbiao
class CustomDataset(DataLoader):
    def __init__(self, dataLoader, transform=None,imageSize=125,batchSize=10):
        self.dataLoader = dataLoader
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.transform = transform
        self.dataCache = np.zeros((len(dataLoader)*batchSize,3,imageSize,imageSize),dtype=np.float32)
        self.labelCache = np.zeros(len(dataLoader)*batchSize,dtype=np.uint8)
        self.inflateCache()

    # 实际加载图片的地方
    def inflateCache(self):
        for i, data in tqdm(enumerate(self.dataLoader, 0)):
            inputDatas=np.zeros((self.batchSize,3,self.imageSize,self.imageSize),dtype=np.float32)
            inputLabels=np.zeros(self.batchSize,dtype=np.uint8)
            for idx,file in enumerate(data['image:FILE'], 0):
                im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, (self.imageSize,self.imageSize))
                image_tensor = Image.fromarray(im)
                normal_img = self.transform(image_tensor)
                inputDatas[idx]=normal_img
                inputLabels[idx]=data['category'][idx]
                idx = idx +1
            self.dataCache[i*self.batchSize:(i+1)*self.batchSize,:,:,:]=inputDatas
            self.labelCache[i*self.batchSize:(i+1)*self.batchSize]=inputLabels
    
    def __len__(self):
        return len(self.dataCache)

    def __getitem__(self, index):
        return self.dataCache[index],self.labelCache[index]