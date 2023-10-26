import torch
from .ResNet18 import ResNet18
import cv2
from PIL import Image
import torch.nn.functional as F

# 本类为工具类
# by wanbiao
class PlantUtil:

    def __init__(self) -> None:
        pass

    # 保存模型
    def saveModel(self,model,name):
        torch.save(model.state_dict(),name)
    # 加载模型
    def loadModel(self,name,numClasses=30) -> ResNet18:
        model = ResNet18.ResNet18(numClasses=numClasses)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(name))
        return model
    
    # 预测图片的工具方法    
    def prePic(self,model,picName,transform,imageSize,top=5):
        im = cv2.imread(picName, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (imageSize,imageSize))
        image = Image.fromarray(im)
        normalImg = transform(image)
        # tensorImage = normalImg.to("cuda" if torch.cuda.is_available() else "cpu")
        tensorImage = normalImg.to("cpu")
        inputImage = torch.zeros(1, 3, imageSize, imageSize).to("cpu")
        inputImage[0]=tensorImage;
        output=model(inputImage)
        outputSoftmax=((F.softmax(output.data,dim=1)*100*100).round())/100
        chineseLabels=[
        '芦荟',
        '香蕉',
        '毛叶阳桃',
        '哈密瓜',
        '木薯',
        '椰子',
        '玉米',
        '黄瓜',
        '姜黄',
        '茄子',
        '高良姜',
        '生姜',
        '番石榴',
        '小青菜',
        '长豆角',
        '芒果',
        '甜瓜',
        '橘子',
        '水稻',
        '番木瓜',
        '辣椒',
        '凤梨',
        '柚子',
        '洋葱',
        '豆子',
        '菠菜',
        '红薯',
        '烟草',
        '水苹果或莲雾',
        '西瓜'
        ]
        values, predicted = torch.topk(outputSoftmax.data, k=top, dim=1)
        values = ((values*10).round())/10
        for input_index in range(len(predicted)):
            preStr = []
            for index in range(top):
                preStr.append(chineseLabels[predicted[input_index][index].item()])
        return values,preStr
        
        