# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
from typing import List, Optional, Union

from modelscope.hub.file_download import model_file_download
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import DEFAULT_MODEL_FOR_PIPELINE, Pipelines
from modelscope.models.base import Model
from modelscope.utils.config import Config, ConfigDict, check_config
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, Invoke,
                                       ModelFile, ThirdParty)
from modelscope.utils.hub import read_config
from modelscope.utils.plugins import (register_modelhub_repo,
                                      register_plugins_repo)
from modelscope.utils.registry import Registry, build_from_cfg
from .base import Pipeline
from .util import is_official_hub_path

PIPELINES = Registry('pipelines')


def normalize_model_input(model,
                          model_revision,
                          third_party=None,
                          ignore_file_pattern=None):
    """ normalize the input model, to ensure that a model str is a valid local path: in other words,
    for model represented by a model id, the model shall be downloaded locally
    """
    if isinstance(model, str) and is_official_hub_path(model, model_revision):
        # skip revision download if model is a local directory
        if not os.path.exists(model):
            # note that if there is already a local copy, snapshot_download will check and skip downloading
            user_agent = {Invoke.KEY: Invoke.PIPELINE}
            if third_party is not None:
                user_agent[ThirdParty.KEY] = third_party
            model = snapshot_download(
                model,
                revision=model_revision,
                user_agent=user_agent,
                ignore_file_pattern=ignore_file_pattern)
    elif isinstance(model, list) and isinstance(model[0], str):
        for idx in range(len(model)):
            if is_official_hub_path(
                    model[idx],
                    model_revision) and not os.path.exists(model[idx]):
                user_agent = {Invoke.KEY: Invoke.PIPELINE}
                if third_party is not None:
                    user_agent[ThirdParty.KEY] = third_party
                model[idx] = snapshot_download(
                    model[idx], revision=model_revision, user_agent=user_agent)
    return model


def build_pipeline(cfg: ConfigDict,
                   task_name: str = None,
                   default_args: dict = None):
    """ build pipeline given model config dict.

    Args:
        cfg (:obj:`ConfigDict`): config dict for model object.
        task_name (str, optional):  task name, refer to
            :obj:`Tasks` for more details.
        default_args (dict, optional): Default initialization arguments.
    """
    return build_from_cfg(
        cfg, PIPELINES, group_key=task_name, default_args=default_args)


def pipeline(task: str = None,
             model: Union[str, List[str], Model, List[Model]] = None,
             preprocessor=None,
             config_file: str = None,
             pipeline_name: str = None,
             framework: str = None,
             device: str = 'gpu',
             model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
             ignore_file_pattern: List[str] = None,
             **kwargs) -> Pipeline:
    """ Factory method to build an obj:`Pipeline`.


    Args:
        task (str): Task name defining which pipeline will be returned.
        model (str or List[str] or obj:`Model` or obj:list[`Model`]): (list of) model name or model object.
        preprocessor: preprocessor object.
        config_file (str, optional): path to config file.
        pipeline_name (str, optional): pipeline class name or alias name.
        framework (str, optional): framework type.
        model_revision: revision of model(s) if getting from model hub, for multiple models, expecting
        all models to have the same revision
        device (str, optional): whether to use gpu or cpu is used to do inference.
        ignore_file_pattern(`str` or `List`, *optional*, default to `None`):
            Any file pattern to be ignored in downloading, like exact file names or file extensions.

    Return:
        pipeline (obj:`Pipeline`): pipeline object for certain task.

    Examples:
        >>> # Using default model for a task
        >>> p = pipeline('image-classification')
        >>> # Using pipeline with a model name
        >>> p = pipeline('text-classification', model='damo/distilbert-base-uncased')
        >>> # Using pipeline with a model object
        >>> resnet = Model.from_pretrained('Resnet')
        >>> p = pipeline('image-classification', model=resnet)
        >>> # Using pipeline with a list of model names
        >>> p = pipeline('audio-kws', model=['damo/audio-tts', 'damo/auto-tts2'])
    """
    if task is None and pipeline_name is None:
        raise ValueError('task or pipeline_name is required')

    third_party = kwargs.get(ThirdParty.KEY)
    if third_party is not None:
        kwargs.pop(ThirdParty.KEY)
    model = normalize_model_input(
        model,
        model_revision,
        third_party=third_party,
        ignore_file_pattern=ignore_file_pattern)
    if pipeline_name is None and kwargs.get('llm_first'):
        pipeline_name = llm_first_checker(model, model_revision)
    pipeline_props = {'type': pipeline_name}
    if pipeline_name is None:
        # get default pipeline for this task
        if isinstance(model, str) \
           or (isinstance(model, list) and isinstance(model[0], str)):
            if is_official_hub_path(model, revision=model_revision):
                # read config file from hub and parse
                cfg = read_config(
                    model, revision=model_revision) if isinstance(
                        model, str) else read_config(
                            model[0], revision=model_revision)
                check_config(cfg)
                register_plugins_repo(cfg.safe_get('plugins'))
                register_modelhub_repo(model, cfg.get('allow_remote', False))
                pipeline_props = cfg.pipeline
        elif model is not None:
            # get pipeline info from Model object
            first_model = model[0] if isinstance(model, list) else model
            if not hasattr(first_model, 'pipeline'):
                # model is instantiated by user, we should parse config again
                cfg = read_config(first_model.model_dir)
                check_config(cfg)
                first_model.pipeline = cfg.pipeline
            pipeline_props = first_model.pipeline
        else:
            pipeline_name, default_model_repo = get_default_pipeline_info(task)
            model = normalize_model_input(default_model_repo, model_revision)
            pipeline_props = {'type': pipeline_name}

    pipeline_props['model'] = model
    pipeline_props['device'] = device
    cfg = ConfigDict(pipeline_props)

    if kwargs:
        cfg.update(kwargs)

    if preprocessor is not None:
        cfg.preprocessor = preprocessor

    return build_pipeline(cfg, task_name=task)


def add_default_pipeline_info(task: str,
                              model_name: str,
                              modelhub_name: str = None,
                              overwrite: bool = False):
    """ Add default model for a task.

    Args:
        task (str): task name.
        model_name (str): model_name.
        modelhub_name (str): name for default modelhub.
        overwrite (bool): overwrite default info.
    """
    if not overwrite:
        assert task not in DEFAULT_MODEL_FOR_PIPELINE, \
            f'task {task} already has default model.'

    DEFAULT_MODEL_FOR_PIPELINE[task] = (model_name, modelhub_name)


def get_default_pipeline_info(task):
    """ Get default info for certain task.

    Args:
        task (str): task name.

    Return:
        A tuple: first element is pipeline name(model_name), second element
            is modelhub name.
    """

    if task not in DEFAULT_MODEL_FOR_PIPELINE:
        # support pipeline which does not register default model
        pipeline_name = list(PIPELINES.modules[task].keys())[0]
        default_model = None
    else:
        pipeline_name, default_model = DEFAULT_MODEL_FOR_PIPELINE[task]
    return pipeline_name, default_model


def llm_first_checker(model: Union[str, List[str], Model, List[Model]],
                      revision: Optional[str]) -> Optional[str]:
    from modelscope.pipelines.nlp.llm_pipeline import LLM_FORMAT_MAP

    def get_file_name(model: str, cfg_name: str,
                      revision: Optional[str]) -> Optional[str]:
        if osp.exists(model):
            return osp.join(model, cfg_name)
        try:
            return model_file_download(model, cfg_name, revision=revision)
        except Exception:
            return None

    def parse_model_type(file: Optional[str], pattern: str) -> Optional[str]:
        if file is None or not osp.exists(file):
            return None
        return Config.from_file(file).safe_get(pattern)

    def get_model_type(model: str, revision: Optional[str]) -> Optional[str]:
        cfg_file = get_file_name(model, ModelFile.CONFIGURATION, revision)
        hf_cfg_file = get_file_name(model, ModelFile.CONFIG, revision)
        cfg_model_type = parse_model_type(cfg_file, 'model.type')
        hf_cfg_model_type = parse_model_type(hf_cfg_file, 'model_type')
        return cfg_model_type or hf_cfg_model_type

    if isinstance(model, list):
        model = model[0]
    if not isinstance(model, str):
        model = model.model_dir
    model_type = get_model_type(model, revision)
    if model_type is not None:
        model_type = model_type.lower().split('-')[0]
        if model_type in LLM_FORMAT_MAP:
            return 'llm'
