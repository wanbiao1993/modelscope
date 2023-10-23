# The implementation is based on resnet, available at https://github.com/biubug6/Pytorch_Retinaface
import torch
import torch.backends.cudnn as cudnn
from .resnet18 import ResNet18


from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(Tasks.plant_classification, module_name=Models.plant_classification)
class PlantClassification(TorchModel):

    def __init__(self, model_path):
        super().__init__(model_path)
        cudnn.benchmark = True
        self.model_path = model_path
        self.net = ResNet18()
        self.load_model()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu" 
        self.net = self.net.to(self.device)

    def load_model(self, load_to_cpu=False):
        pretrained_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        return self.net(input)
