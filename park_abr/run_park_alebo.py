# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import json

import numpy as np
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.storage.json_store.encoder import object_to_json
from ax.service.ax_client import AxClient

from fb_abr_problem import ParkContextualRunner

########
# Define problem
ABR_CONTEXT_CONFIG_DICT = {
    'c0': {'name': 'c0', 'delay': 0.09111558887847584},
    'c1': {'name': 'c1', 'delay': 0.13919983731019495},
    'c10': {'name': 'c10', 'delay': 0.04709563378153773},
    'c11': {'name': 'c11', 'delay': 0.09175980911983045},
    'c2': {'name': 'c2', 'delay': 0.05811786663939401},
    'c3': {'name': 'c3', 'delay': 0.15680707174733982},
    'c4': {'name': 'c4', 'delay': 0.21008791350238118},
    'c5': {'name': 'c5', 'delay': 0.12895778597785987},
    'c6': {'name': 'c6', 'delay': 0.05922074675831855},
    'c7': {'name': 'c7', 'delay': 0.0751735817104147},
    'c8': {'name': 'c8', 'delay': 0.08200189263592551},
    'c9': {'name': 'c9', 'delay': 0.0962324885998359}
}
num_trials = 75

for rep in range(25):
    print('====================', rep)

    num_contexts = len(ABR_CONTEXT_CONFIG_DICT)
    benchmark_problem = ParkContextualRunner(
        num_contexts=num_contexts,
        context_dict=ABR_CONTEXT_CONFIG_DICT
    )
    decomposition = benchmark_problem.contextual_parameter_decomposition

    t1 = time.time()

    gs = ALEBOStrategy(
        D=len(benchmark_problem._contextual_parameters), d=8, init_size=8
    )
    axc = AxClient(generation_strategy=gs)

    experiment_parameters = benchmark_problem.contextual_parameters
    axc.create_experiment(
        name="cbo_aggregated_reward_experiment",
        parameters=experiment_parameters,
        objective_name="aggregated_reward",
        minimize=True,
    )
    context_reward_list = []

    def evaluation_aggregated_reward(parameters):
        # put parameters into 1-D array
        # x = [bw_c0, bf_c0, c_c0, ...]
        x = []
        for context_name in benchmark_problem.context_name_list:
            x.extend([parameters.get(param) for param in decomposition[context_name]])
        aggregated_reward, context_reward = benchmark_problem.f(np.array(x))
        return {"aggregated_reward": (aggregated_reward, 0.0)}, context_reward

    for itrial in range(num_trials):
        parameters, trial_index = axc.get_next_trial()
        aggregated_res, context_res = evaluation_aggregated_reward(parameters)
        axc.complete_trial(trial_index=trial_index, raw_data=aggregated_res)
        context_res["trial_index"] = itrial
        context_reward_list.append(context_res)

    res = json.dumps(
        {
            "experiment": object_to_json(axc.experiment),
            "context_rewards": context_reward_list,
        }
    )
    with open(f'results/alebo_park_rep_{rep}.json', "w") as fout:
       json.dump(res, fout)

    print ('=============', time.time() - t1)
