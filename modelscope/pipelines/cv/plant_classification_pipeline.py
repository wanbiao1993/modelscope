# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from torchvision import transforms,models

from modelscope.metainfo import Pipelines
from modelscope.models.cv.plant_classification.plantclassification import PlantClassification
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.plant_classification, module_name=Pipelines.plant_classification)
class PlantClassificationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a plant classification pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        ckpt_path = osp.join(model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {ckpt_path}')
        detector = PlantClassification(model_path=ckpt_path)
        self.detector = detector
        logger.info('load model done from fudan university by wanbiao')

    def preprocess(self, input: Input):
        img_raw = LoadImage.convert_to_ndarray(input)
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # img = Image.open(input) 
        img_tfm = tfm(img_raw)

        img=img_tfm.to("cuda" if torch.cuda.is_available() else "cpu") 

        img2 = img.unsqueeze(0)
        return img2
    
    def forward(self, input):
        result = self.detector(input)
        assert result is not None
        return result
    
    def postprocess(self, inputs):
        return inputs