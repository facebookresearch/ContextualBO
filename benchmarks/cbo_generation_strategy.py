#! /usr/bin/env python3
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
from ax.models.torch.cbo_lcem import LCEMBO
from ax.modelbridge.factory import DEFAULT_TORCH_DEVICE
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import (
    Cont_X_trans,
    StratifiedStandardizeY,
    TaskEncode,
    Y_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.random.sobol import SobolGenerator
from ax.models.torch.botorch import BotorchModel


def get_multisobol(search_space: SearchSpace) -> RandomModelBridge:
    return RandomModelBridge(
        search_space=search_space,
        model=SobolGenerator(),
        transforms=[TaskEncode] + Cont_X_trans,
    )


class MultiSOBOLStrategy(GenerationStrategy):
    def __init__(
        self, context_parameter: ChoiceParameter, name: str = "MultiSOBOL"
    ) -> None:
        self.context_parameter = context_parameter
        self.num_contexts = len(context_parameter.values)
        steps = [GenerationStep(get_multisobol, -1)]
        super().__init__(steps=steps, name=name)

    def clone_reset(self) -> "MultiSOBOLStrategy":
        """Copy without state."""
        return self.__class__(context_parameter=self.context_parameter, name=self.name)

    def gen(
        self,
        experiment: Experiment,
        data: Optional[Data] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> GeneratorRun:
        """Produce the next points in the experiment."""
        num_trials = len(self._generator_runs)
        idx = num_trials % self.num_contexts  # decide which context to optimize
        fixed_features = ObservationFeatures(
            parameters={self.context_parameter.name: self.context_parameter.values[idx]}
        )
        generator_run = super().gen(
            experiment=experiment, data=data, n=1, fixed_features=fixed_features
        )
        return generator_run


def get_multioutput(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    status_quo_features: Optional[ObservationFeatures] = None,
) -> TorchModelBridge:
    # Set transforms for a Single-type MTGP model.
    transforms = Cont_X_trans + [StratifiedStandardizeY, TaskEncode]
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space or experiment.search_space,
        data=data,
        model=BotorchModel(),
        transforms=transforms,
        torch_dtype=torch.double,
        status_quo_features=status_quo_features,
    )


class MultiOutputStrategy(GenerationStrategy):
    def __init__(
        self,
        context_parameter: ChoiceParameter,
        init_size: int,
        steps: Optional[List[GenerationStep]] = None,
        name: str = "MultiOutputBO",
    ) -> None:
        self.context_parameter = context_parameter
        self.num_contexts = len(context_parameter.values)
        if steps is None:
            steps = [
                GenerationStep(get_multisobol, init_size),
                GenerationStep(get_multioutput, -1),
            ]
        super().__init__(steps=steps, name=name)

    def clone_reset(self) -> "MultiOutputStrategy":
        """Copy without state."""
        return self.__class__(context_parameter=self.context_parameter, name=self.name)

    def gen(
        self,
        experiment: Experiment,
        data: Optional[Data] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> GeneratorRun:
        """Produce the next points in the experiment."""
        num_trials = len(self._generator_runs)
        idx = num_trials % self.num_contexts  # decide which context to optimize
        fixed_features = ObservationFeatures(
            parameters={self.context_parameter.name: self.context_parameter.values[idx]}
        )
        generator_run = super().gen(
            experiment=experiment, data=data, n=1, fixed_features=fixed_features
        )
        return generator_run


def get_multitask_contextualBO(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    status_quo_features: Optional[ObservationFeatures] = None,
) -> TorchModelBridge:
    # Set transforms for a Single-type MTGP model.
    transforms = Cont_X_trans + [StratifiedStandardizeY, TaskEncode]
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space or experiment.search_space,
        data=data,
        model=LCEMBO(),
        transforms=transforms,
        torch_dtype=torch.double,
        status_quo_features=status_quo_features,
    )


class MultiTaskContextualBOStrategy(MultiOutputStrategy):
    def __init__(
        self,
        context_parameter: ChoiceParameter,
        init_size: int,
        name: str = "MultiTaskContextualBO",
    ) -> None:
        steps = [
            GenerationStep(get_multisobol, init_size),
            GenerationStep(get_multitask_contextualBO, -1),
        ]
        super().__init__(
            context_parameter=context_parameter,
            init_size=init_size,
            steps=steps,
            name=name,
        )


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
