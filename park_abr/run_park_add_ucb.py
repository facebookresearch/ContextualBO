# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run benchmarks for Add-GP-UCB.

Requires installing dragonfly-opt from pip. The experiments here used version
0.1.4.
"""
import cma

import time
import json

from argparse import Namespace
import numpy as np
from dragonfly import minimise_function

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

    r_contextual = ParkContextualRunner(
        num_contexts=len(ABR_CONTEXT_CONFIG_DICT),
        context_dict=ABR_CONTEXT_CONFIG_DICT,
        max_eval=75,
        return_context_reward=False,
    )

    t1 = time.time()

    options = Namespace(acq="add_ucb")
    try:
        minimise_function(
            r_contextual.f,
            domain=[[0.0, 1.0], [0.0, 3.0], [0.0, 1.0], [0.0001, 0.25]] * len(ABR_CONTEXT_CONFIG_DICT),
            max_capital=num_trials,
            options=options,
        )
    except StopIteration:
        pass

    with open(f'results/add_ucb_park_rep_{rep}.json', 'w') as fout:
        # times -1 to fs
        json.dump([-reward for reward in r_contextual.fs], fout)

    print ('=============', time.time() - t1)
