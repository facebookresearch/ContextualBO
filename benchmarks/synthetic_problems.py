#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC
from typing import Dict, List

import numpy as np
import torch
from ax.models.random.sobol import SobolGenerator
from ax.utils.measurement.synthetic_functions import branin, hartmann6


class ContextualSyntheticFunction(ABC):
    def __init__(
        self,
        context_name_list: List,
        context_weights: np.ndarray,
        noise_sd: float = 0.0,
    ) -> None:
        self.context_name_list = context_name_list
        # contextual weights
        self.context_weights = context_weights
        # number of contexts
        self.num_contexts = len(context_name_list)
        # noise term
        self.noise_sd = noise_sd
        self._base_parameters = []
        self._contextual_parameters = []
        self._decomposition = {}

    @property
    def base_parameters(self) -> List[Dict]:
        return self._base_parameters

    @property
    def contextual_parameters(self) -> List[Dict]:
        return self._contextual_parameters

    @property
    def contextual_parameter_decomposition(self) -> List[Dict]:
        return self._decomposition

    def component_function(self, x: np.ndarray) -> float:
        """function that produces the outcomes for each component."""
        raise NotImplementedError

    def evaluation_function_vectorized(self, x: np.ndarray) -> np.ndarray:
        # input x is a matrix: each row corresponds each context
        # and each column to each base parameter
        return np.array(
            [self.component_function(x[i, :]) for i in range(self.num_contexts)]
        )

    def evaluation_function_aggregated(self, x: np.ndarray) -> float:
        # aggregate across context weighted by context coeff
        context_output = self.evaluation_function_vectorized(x)
        return np.sum(context_output * self.context_weights)


class ContextualEmbeddingSyntheticFunction(ContextualSyntheticFunction):
    def __init__(
        self,
        context_name_list: List,
        context_weights: np.ndarray,
        context_embedding: np.ndarray,
        noise_sd: float = 0.0,
    ) -> None:
        super().__init__(
            context_name_list=context_name_list,
            context_weights=context_weights,
            noise_sd=noise_sd,
        )
        # each row corresponds each context and each column to each embeddding
        self.context_embedding = context_embedding

    def evaluation_function_vectorized(self, x: np.ndarray) -> np.ndarray:
        # input x is a matrix: each row corresponds each context
        # and each column to each base parameter
        x_all = np.hstack([x, self.context_embedding])
        return np.array(
            [self.component_function(x_all[i, :]) for i in range(self.num_contexts)]
        )


class Branin2DBase(ContextualSyntheticFunction):
    def __init__(
        self,
        context_name_list: List,
        context_weights: np.ndarray,
        noise_sd: float = 0.0,
    ) -> None:
        super().__init__(
            context_name_list=context_name_list,
            context_weights=context_weights,
            noise_sd=noise_sd,
        )
        # define search space for non-dp setting
        self._base_parameters = [
            {
                "name": "x0",
                "type": "range",
                "bounds": [-5.0, 10.0],
                "value_type": "float",
                "log_scale": False,
            },
            {
                "name": "x1",
                "type": "range",
                "bounds": [0.0, 15.0],
                "value_type": "float",
                "log_scale": False,
            },
        ]

        # for dp setting, extend to contextual search space
        self._contextual_parameters = []
        for context_name in self.context_name_list:
            self._contextual_parameters.append(
                {
                    "name": f"x0_{context_name}",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                    "value_type": "float",
                    "log_scale": False,
                }
            )
            self._contextual_parameters.append(
                {
                    "name": f"x1_{context_name}",
                    "type": "range",
                    "bounds": [0.0, 15.0],
                    "value_type": "float",
                    "log_scale": False,
                }
            )
        self._decomposition = {
            f"{context_name}": [f"x0_{context_name}", f"x1_{context_name}"]
            for context_name in self.context_name_list
        }

    def component_function(self, x: np.ndarray) -> float:
        return branin.f(x)


class Branin1DEmbedding(ContextualEmbeddingSyntheticFunction):
    def __init__(
        self,
        context_name_list: List,
        context_weights: np.ndarray,
        context_embedding: np.ndarray,
        noise_sd: float = 0.0,
    ) -> None:
        super().__init__(
            context_name_list=context_name_list,
            context_weights=context_weights,
            context_embedding=context_embedding,  # values between [0.0, 15.0]
            noise_sd=noise_sd,
        )

        # define search space for non-dp setting
        self._base_parameters = [
            {
                "name": "x0",
                "type": "range",
                "bounds": [-5.0, 10.0],
                "value_type": "float",
                "log_scale": False,
            }
        ]
        # for dp setting, extend to contextual search space
        self._contextual_parameters = []
        for context_name in self.context_name_list:
            self._contextual_parameters.append(
                {
                    "name": f"x0_{context_name}",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                    "value_type": "float",
                    "log_scale": False,
                }
            )
        self._decomposition = {
            f"{context_name}": [f"x0_{context_name}"]
            for context_name in self.context_name_list
        }

    def component_function(self, x: np.ndarray) -> float:
        # make sure embedding is at the end of the array
        return branin.f(x)


class Hartmann6DBase(ContextualSyntheticFunction):
    # additive brannin 2d case
    def __init__(
        self,
        context_name_list: List,
        context_weights: np.ndarray,
        noise_sd: float = 0.0,
    ) -> None:
        super().__init__(
            context_name_list=context_name_list,
            context_weights=context_weights,
            noise_sd=noise_sd,
        )

        # define search space for non-dp setting
        self._base_parameters = [
            {
                "name": f"x{i}",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float",
                "log_scale": False,
            }
            for i in range(6)
        ]

        # for dp setting, extend to contextual search space
        self._contextual_parameters = []
        for context_name in self.context_name_list:
            self._contextual_parameters.extend(
                [
                    {
                        "name": f"x{j}_{context_name}",
                        "type": "range",
                        "bounds": [0.0, 1.0],
                        "value_type": "float",
                        "log_scale": False,
                    }
                    for j in range(6)
                ]
            )

        self._decomposition = {
            f"{context_name}": [f"x{j}_{context_name}" for j in range(6)]
            for context_name in self.context_name_list
        }

    def component_function(self, x: np.ndarray) -> float:
        return hartmann6.f(x)


class Hartmann5DEmbedding(ContextualEmbeddingSyntheticFunction):
    def __init__(
        self,
        context_name_list: List,
        context_weights: np.ndarray,
        context_embedding: np.ndarray,
        noise_sd: float = 0.0,
    ) -> None:
        super().__init__(
            context_name_list=context_name_list,
            context_weights=context_weights,
            context_embedding=context_embedding,  # values between [0.0, 1.0]
            noise_sd=noise_sd,
        )

        # define search space for non-dp setting
        self._base_parameters = [
            {
                "name": f"x{i}",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float",
                "log_scale": False,
            }
            for i in range(5)
        ]

        # for dp setting, extend to contextual search space
        self._contextual_parameters = []
        for context_name in self.context_name_list:
            self._contextual_parameters.extend(
                [
                    {
                        "name": f"x{j}_{context_name}",
                        "type": "range",
                        "bounds": [0.0, 1.0],
                        "value_type": "float",
                        "log_scale": False,
                    }
                    for j in range(5)
                ]
            )

        self._decomposition = {
            f"{context_name}": [f"x{j}_{context_name}" for j in range(5)]
            for context_name in self.context_name_list
        }

    def component_function(self, x: np.ndarray) -> float:
        # make sure embedding is at the end of the array
        return hartmann6.f(x)
