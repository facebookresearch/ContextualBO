# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import numpy as np
from synthetic_problems import (
    Branin1DEmbedding,
    Branin2DBase,
    Hartmann5DEmbedding,
    Hartmann6DBase,
)


def get_benchmark_problem(
    name: str,
    num_contexts: int,
    benchmark_problem_args: Optional[Dict[str, Any]] = None,
):
    """generate benchmark problems.
    Args:
        1. name: benchmark name
        2. num_contexts: number of contexts n
        3. args for creating benchmark
            - context_name_list. List of str. Default is [c0, c1, ..., cn]
            - context_weights. [w0, w1, ..., wn]. sum of w_i = 1. Default is [1/n]
            - context_embedding.
    """
    benchmark_problem_args = benchmark_problem_args or {}
    context_name_list = benchmark_problem_args.get(
        "context_name_list", [f"c{i}" for i in range(num_contexts)]
    )
    context_weights = np.array(
        benchmark_problem_args.get(
            "context_weights", np.ones(num_contexts) / num_contexts
        )
    )
    if name == "Branin2D":
        benchmark_problem = Branin2DBase(
            context_name_list=context_name_list, context_weights=context_weights
        )
    elif name == "Hartmann6D":
        benchmark_problem = Hartmann6DBase(
            context_name_list=context_name_list, context_weights=context_weights
        )
    elif name == "Branin1DEmbedding":
        benchmark_problem = Branin1DEmbedding(
            context_name_list=context_name_list,
            context_weights=context_weights,
            context_embedding=np.arange(0.0, 15.0, 15.0 / num_contexts).reshape(
                num_contexts, 1
            ),
        )
    elif name == "Hartmann5DEmbedding":
        context_embedding = np.array(
            benchmark_problem_args.get(
                "context_embedding", np.arange(0.0, 1.0, 1.0 / num_contexts)
            )
        )
        benchmark_problem = Hartmann5DEmbedding(
            context_name_list=context_name_list,
            context_weights=context_weights,
            context_embedding=context_embedding.reshape(num_contexts, 1),
        )
    return benchmark_problem
