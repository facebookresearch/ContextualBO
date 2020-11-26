#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter
from ax.core.search_space import SearchSpace
from ax.models.torch.cbo_lcea import LCEABO
from ax.models.torch.cbo_sac import SACBO
from ax.modelbridge.factory import DEFAULT_TORCH_DEVICE
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import (
    Cont_X_trans,
    Y_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.random.sobol import SobolGenerator
from ax.models.torch.botorch import BotorchModel


def get_ContextualBO(
    experiment: Experiment,
    data: Data,
    decomposition: Dict[str, List[str]],
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: List[Type[Transform]] = Cont_X_trans + Y_trans,
) -> TorchModelBridge:
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space or experiment.search_space,
        data=data,
        model=SACBO(decomposition=decomposition),
        transforms=transforms,
        torch_dtype=dtype,
        torch_device=device,
    )


def get_ContextualEmbeddingBO(
    experiment: Experiment,
    data: Data,
    decomposition: Dict[str, List[str]],
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: List[Type[Transform]] = Cont_X_trans + Y_trans,
    cat_feature_dict: Optional[Dict] = None,
    embs_feature_dict: Optional[Dict] = None,
    context_weight_dict: Optional[Dict] = None,
    embs_dim_list: Optional[List[int]] = None,
    search_space: Optional[SearchSpace] = None,
    gp_model_args: Optional[Dict[str, Any]] = None,
) -> TorchModelBridge:
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space or experiment.search_space,
        data=data,
        model=LCEABO(
            decomposition=decomposition,
            cat_feature_dict=cat_feature_dict,
            embs_feature_dict=embs_feature_dict,
            context_weight_dict=context_weight_dict,
            embs_dim_list=embs_dim_list,
            gp_model_args=gp_model_args,
        ),
        transforms=transforms,
        torch_dtype=dtype,
        torch_device=device,
    )
