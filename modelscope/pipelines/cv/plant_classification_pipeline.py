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
        detector = PlantClassification(model_dir=model)
        self.detector = detector
        logger.info('load model done from fudan university by wanbiao')

    def preprocess(self, input: Input):
        return input
    
    def forward(self, input):
        result = self.detector(input)
        assert result is not None
        return result
    
    def postprocess(self, inputs):
        return inputs