#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import numpy as np
from ax.modelbridge.factory import get_GPEI, get_sobol
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient
from ax.storage.json_store.encoder import object_to_json
from cbo_generation_strategy import (
    get_ContextualBO,
    get_ContextualEmbeddingBO,
)
from get_synthetic_problem import get_benchmark_problem


CBO_EMB_MODEL_GEN_OPTIONS = {
    "acquisition_function_kwargs": {"q": 1, "noiseless": True},
    "optimizer_kwargs": {
        "method": "SLSQP",
        "batch_limit": 1,
        "joint_optimization": True,
    },
}


def run_aggregated_reward_benchmark(
    strategy_name,
    benchmark_problem_name,
    irep,
    num_contexts,
    init_size=8,
    num_trials=100,
    benchmark_problem_args={},
):
    benchmark_problem = get_benchmark_problem(
        name=benchmark_problem_name,
        num_contexts=num_contexts,
        benchmark_problem_args=benchmark_problem_args,
    )
    decomposition = benchmark_problem.contextual_parameter_decomposition

    context_weight_dict = {
        benchmark_problem.context_name_list[i]: benchmark_problem.context_weights[i]
        for i in range(benchmark_problem.num_contexts)
    }
    embs_feature_dict = {
        benchmark_problem.context_name_list[i]: benchmark_problem.context_embedding[
            i, :
        ]
        for i in range(benchmark_problem.num_contexts)
    }

    if strategy_name == "Sobol":
        gs = GenerationStrategy(
            name="Sobol", steps=[GenerationStep(get_sobol, -1)]
        )
    elif strategy_name == "GPEI":
        gs = GenerationStrategy(
            name="GPEI",
            steps=[
                GenerationStep(get_sobol, init_size),
                GenerationStep(get_GPEI, -1),
            ],
        )
    elif strategy_name == "SAC":
        gs = GenerationStrategy(
            name="SAC",
            steps=[
                GenerationStep(model=get_sobol, num_trials=init_size),
                GenerationStep(
                    model=get_ContextualBO,
                    num_trials=-1,
                    model_kwargs={"decomposition": decomposition},
                ),
            ],
        )
    elif strategy_name == "LCE-A":
        gs = GenerationStrategy(
            name="LCE-A",
            steps=[
                GenerationStep(model=get_sobol, num_trials=init_size),
                GenerationStep(
                    model=get_ContextualEmbeddingBO,
                    num_trials=-1,
                    model_kwargs={
                        "decomposition": decomposition,
                    },
                    model_gen_kwargs={"model_gen_options": CBO_EMB_MODEL_GEN_OPTIONS},
                ),
            ],
        )

    axc = AxClient(generation_strategy=gs)

    experiment_parameters = benchmark_problem.contextual_parameters
    axc.create_experiment(
        name="cbo_aggregated_reward_experiment",
        parameters=experiment_parameters,
        objective_name="aggregated_reward",
        minimize=True,
    )

    def evaluation_aggregated_reward(parameters):
        # put parameters into 2-D array
        x = np.array(
            [
                [parameters.get(param) for param in decomposition[context_name]]
                for context_name in benchmark_problem.context_name_list
            ]
        )
        return {
            "aggregated_reward": (
                benchmark_problem.evaluation_function_aggregated(x),
                0.0,
            )
        }

    for _ in range(num_trials):
        parameters, trial_index = axc.get_next_trial()
        axc.complete_trial(
            trial_index=trial_index, raw_data=evaluation_aggregated_reward(parameters)
        )

    res = json.dumps(object_to_json(axc.experiment))
    with open(f'results/aggregated_reward_{benchmark_problem_name}_{strategy}_rep_{irep}.json', "w") as fout:
       json.dump(res, fout)
    return res


def run_aggregated_reward_benchmark_reps(
    benchmark_problem_name,
    strategy,
    num_contexts,
    init_size=8,
    num_trials=100,
    reps=8,
    benchmark_problem_args={},
):
    res = {strategy: []}

    for irep in range(reps):
        res[strategy].append(
            run_aggregated_reward_benchmark(
                strategy_name=strategy,
                benchmark_problem_name=benchmark_problem_name,
                irep=irep,
                num_contexts=num_contexts,
                init_size=init_size,
                num_trials=num_trials,
                benchmark_problem_args=benchmark_problem_args,
            )
        )
    with open(f'results/aggregated_reward_{benchmark_problem_name}_{strategy}.json', "w") as fout:
        json.dump(res, fout)


if __name__ == '__main__':
    # Run all of the benchmark replicates.

    # Hartmann5DEmbedding, Uniform Weights, SAC
    # run_aggregated_reward_benchmark_reps(
    #     benchmark_problem_name="Hartmann5DEmbedding",
    #     strategy="CBO",
    #     num_contexts=5,
    #     reps=10
    # )

    # # Hartmann5DEmbedding, Uniform Weights, LCE-A
    # run_aggregated_reward_benchmark_reps(
    #     benchmark_problem_name="Hartmann5DEmbedding",
    #     strategy="CBO_Emb_MAP",
    #     num_contexts=5,
    #     reps=10
    # )

    # Hartmann5DEmbedding, Uniform Weights, SOBOL
    run_aggregated_reward_benchmark_reps(
        benchmark_problem_name="Hartmann5DEmbedding",
        strategy="Sobol",
        num_contexts=5,
        num_trials=10,
        reps=1
    )

    # Hartmann5DEmbedding, Uniform Weights, Standard BO
    run_aggregated_reward_benchmark_reps(
        benchmark_problem_name="Hartmann5DEmbedding",
        strategy="GPEI",
        num_contexts=5,
        num_trials=10,
        reps=1
    )

    # Hartmann5DEmbedding, Skewed Weights, num of contexts = 10, num of dense contexts = 2
    # run_aggregated_reward_benchmark_reps(
    #     benchmark_problem_name="Hartmann5DEmbedding",
    #     strategy="CBO",
    #     num_contexts=10,
    #     reps=10,
    #     benchmark_problem_args = {"context_weights": [0.46, 0.46, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]}
    # )
