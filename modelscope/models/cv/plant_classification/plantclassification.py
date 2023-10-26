# The implementation is based on resnet, available at https://github.com/biubug6/Pytorch_Retinaface
import torch
import os
import torch.backends.cudnn as cudnn
from .ResNet18 import ResNet18
import mmcv
from .Utils import PlantUtil
import torchvision.transforms as transforms


from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks

@MODELS.register_module(Tasks.plant_classification, module_name=Models.plant_classification)
class PlantClassification(TorchModel): 

    def __init__(self, model_dir: str, **kwargs):

        super().__init__(model_dir)
        self.ms_model_dir = model_dir

        self.cls_model = ResNet18(30)
        self.load_model()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu" 
        self.cls_model = self.cls_model.to(self.device)

    def load_model(self, load_to_cpu=False):
        checkpoint_path = os.path.join(self.ms_model_dir,ModelFile.TORCH_MODEL_FILE)
        pretrained_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.cls_model.load_state_dict(pretrained_dict, strict=False)
        self.cls_model.eval()
        self.CLASSES = [
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
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        msDataTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        print("input----")
        print(input)
        values,result = PlantUtil().prePic(self.cls_model,input,msDataTransform,150,5)
        return values,result
