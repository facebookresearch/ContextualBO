#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
from typing import Any, Dict, List, Optional

import numpy as np
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.service.ax_client import AxClient
from ax.storage.json_store.encoder import object_to_json
from get_synthetic_problem import get_benchmark_problem
from cbo_generation_strategy import (
    MultiOutputStrategy,
    MultiSOBOLStrategy,
    MultiTaskContextualBOStrategy,
)
from synthetic_problems import ContextualEmbeddingSyntheticFunction


def run_multioutput_reward_benchmark(
    strategy_name,
    benchmark_problem_name,
    irep,
    num_contexts,
    init_size=8,
    num_trials=100,
    benchmark_problem_args={},
):
    init_size = init_size * num_contexts
    num_trials = num_trials * num_contexts

    benchmark_problem = get_benchmark_problem(
        name=benchmark_problem_name,
        num_contexts=num_contexts,
        benchmark_problem_args=benchmark_problem_args,
    )
    context_parameter = ChoiceParameter(
        name="CONTEXT_PARAMS",
        values=benchmark_problem.context_name_list,
        is_task=True,
        parameter_type=ParameterType.STRING,
    )

    if strategy_name == "MultiSOBOL":
        gs = MultiSOBOLStrategy(context_parameter=context_parameter, name="MultiSOBOL")
    elif strategy_name == "ICM":
        gs = MultiOutputStrategy(
            name="ICM",
            context_parameter=context_parameter,
            init_size=init_size,
        )
    elif strategy_name == "LCE-M":
        gs = MultiTaskContextualBOStrategy(
            name="LCE-M",
            context_parameter=context_parameter,
            init_size=init_size,
        )

    axc = AxClient(generation_strategy=gs)

    experiment_parameters = benchmark_problem.base_parameters
    experiment_parameters.append(
        {
            "name": context_parameter.name,
            "type": "choice",
            "values": context_parameter.values,
            "is_task": True,
        }
    )

    axc.create_experiment(
        name="cbo_multioutput_reward_experiment",
        parameters=experiment_parameters,
        objective_name="context_reward",
        minimize=True,
    )

    def evaluation_contextual_reward(parameters):
        # get base parameters only (into 1-D array)
        x = np.array(
            [
                parameters.get(param["name"])
                for param in benchmark_problem.base_parameters
                if param["name"] != context_parameter.name
            ]
        )
        context = parameters.get(context_parameter.name)
        weight = benchmark_problem.context_weights[
            benchmark_problem.context_name_list.index(context)
        ]
        if isinstance(benchmark_problem, ContextualEmbeddingSyntheticFunction):
            embs = benchmark_problem.context_embedding[
                benchmark_problem.context_name_list.index(context), :
            ]
            x = np.hstack([x, embs])
        return {
            "context_reward": (weight * benchmark_problem.component_function(x), 0.0)
        }

    for _ in range(num_trials):
        parameters, trial_index = axc.get_next_trial()
        axc.complete_trial(
            trial_index=trial_index, raw_data=evaluation_contextual_reward(parameters)
        )

    res = json.dumps(object_to_json(axc.experiment))
    with open(f'results/multioutput_reward_{benchmark_problem_name}_{strategy_name}_rep_{irep}.json', "w") as fout:
       json.dump(res, fout)
    return res


def run_multioutput_reward_benchmark_reps(
    benchmark_problem_name,
    strategy,
    num_contexts,
    init_size=4,
    num_trials=40,
    reps=8,
    benchmark_problem_args={},
):
    res = {strategy: []}

    for irep in range(reps):
        res[strategy].append(
            run_multioutput_reward_benchmark(
                strategy_name=strategy,
                benchmark_problem_name=benchmark_problem_name,
                irep=irep,
                num_contexts=num_contexts,
                init_size=init_size,
                num_trials=num_trials,
                benchmark_problem_args=benchmark_problem_args,
            )
        )
    with open(f'results/multioutput_reward_{benchmark_problem_name}_{strategy}.json', "w") as fout:
       json.dump(res, fout)


if __name__ == '__main__':
    # Run all of the benchmark replicates.

    # Hartmann5DEmbedding, Uniform Weights, LCE-M
    run_multioutput_reward_benchmark_reps(
        benchmark_problem_name="Hartmann5DEmbedding",
        strategy="LCE-M",
        num_contexts=5,
        reps=8
    )

    # Hartmann5DEmbedding, Uniform Weights, ICM
    run_multioutput_reward_benchmark_reps(
        benchmark_problem_name="Hartmann5DEmbedding",
        strategy="ICM",
        num_contexts=5,
        reps=8
    )

    # Hartmann5DEmbedding, Uniform Weights, SOBOL
    run_multioutput_reward_benchmark_reps(
        benchmark_problem_name="Hartmann5DEmbedding",
        strategy="MultiSOBOL",
        num_contexts=5,
        reps=8
    )
