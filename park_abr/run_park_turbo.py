# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run benchmarks for TuRBO.

Requires installing turbo from https://github.com/uber-research/TuRBO.
"""
import turbo

import time
import json

import numpy.matlib
import numpy as np


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

lbs = np.array([0.0, 0.0, 0.0, 0.0001] * len(ABR_CONTEXT_CONFIG_DICT))
ubs = np.array([1.0, 3.0, 1.0, 0.25] * len(ABR_CONTEXT_CONFIG_DICT))

for rep in range(25):
    print('====================', rep)

    r_contextual = ParkContextualRunner(
        num_contexts=len(ABR_CONTEXT_CONFIG_DICT),
        context_dict=ABR_CONTEXT_CONFIG_DICT,
        max_eval=75,
        return_context_reward=False,
    )
    t1 = time.time()
    turbo1 = turbo.Turbo1(
        f=r_contextual.f,
        lb=lbs,
        ub=ubs,
        n_init=8,
        max_evals=75,
    )
    turbo1.optimize()

    with open(f'results/turbo_park_rep_{rep}.json', 'w') as fout:
        # times -1 to fs
        json.dump([-reward for reward in r_contextual.fs], fout)
        # json.dump(r_contextual.fs, fout)
    print ('=============', time.time() - t1)
