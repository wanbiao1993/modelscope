# Copyright (c) Alibaba, Inc. and its affiliates.
import io
from typing import Any, Dict, Union

import cv2
import numpy as np
import PIL
from numpy import ndarray
from PIL import Image, ImageOps

from modelscope.metainfo import Preprocessors
from modelscope.utils.constant import Fields
from .base import Preprocessor
from .builder import PREPROCESSORS

@PREPROCESSORS.register_module(
    Fields.cv,
    module_name=Preprocessors.plant_classification_preprocessor)
class PlantClassificationPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        """image classification bypass preprocessor in the fine-tune scenario
        """
        super().__init__(*args, **kwargs)
        self.preprocessor_val_cfg = kwargs.pop('val', None)

    def eval(self):
        self.training = False
        return

    def __call__(self, results: Dict[str, Any]):
        """process the raw input data

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            Dict[str, Any] | None: the preprocessed data
        """
        pass
