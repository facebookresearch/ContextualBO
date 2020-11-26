# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run benchmarks for Ensemble BO.

Requires installing EBO from https://github.com/zi-w/Ensemble-Bayesian-Optimization.
"""
import os
import sys
sys.path.insert(1, os.path.join(os.getcwd(), 'Ensemble-Bayesian-Optimization'))

import time
import json

from ebo_core.ebo import ebo
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

###### Prepare EBO

# All options except: x_range, dx, max_value, T, gp_sigma, dim_limit (3)
# Taken as package defaults from test_ebo.py
core_options = {
    'B':10, # number of candidates to be evaluated
    'dim_limit':4, # max dimension of the input for each additive function component
    'isplot':0, # 1 if plotting the result; otherwise 0.
    'z':None, 'k':None, # group assignment and number of cuts in the Gibbs sampling subroutine
    'alpha':1., # hyperparameter of the Gibbs sampling subroutine
    'beta':np.array([5.,2.]),
    'opt_n':1000, # points randomly sampled to start continuous optimization of acfun
    'pid':'test3', # process ID for Azure
    'datadir':'tmp_data/', # temporary data directory for Azure
    'gibbs_iter':10, # number of iterations for the Gibbs sampling subroutine
    'useAzure':False, # set to True if use Azure for batch evaluation
    'func_cheap':True, # if func cheap, we do not use Azure to test functions
    'n_add':None, # this should always be None. it makes dim_limit complicated if not None.
    'nlayers': 100, # number of the layers of tiles
    'gp_type':'l1', # other choices are l1, sk, sf, dk, df
    'n_bo':10, # min number of points selected for each partition
    'n_bo_top_percent': 0.5, # percentage of top in bo selections
    'n_top':10, # how many points to look ahead when doing choose Xnew
    'min_leaf_size':10, # min number of samples in each leaf
    'max_n_leaves':10, # max number of leaves
    'thresAzure':1, # if batch size > thresAzure, we use Azure
    'save_file_name': 'tmp/tmp.pk',
}

for rep in range(25):
    print('================', rep)
    options = {
        'x_range': np.vstack((lbs, ubs)),
        'dx': 4 * len(ABR_CONTEXT_CONFIG_DICT),
        'max_value': 180,  # Give it a pretty good guess for max value
        'T': 75,
        'gp_sigma': 1e-7,
    }
    options.update(core_options)

    ##### Run optimization
    r_contextual = ParkContextualRunner(
        num_contexts=len(ABR_CONTEXT_CONFIG_DICT),
        context_dict=ABR_CONTEXT_CONFIG_DICT,
        max_eval=75,
        return_context_reward=False,
    )

    t1 = time.time()
    f = lambda x: -r_contextual.f(x)  # since EBO maximizes
    e = ebo(f, options)
    try:
        e.run()
    except Exception:
        pass


    with open("results/ebo_park_rep_{rep}.json".format(rep), 'w') as fout:
        # times -1 to fs
        json.dump([-reward for reward in r_contextual.fs], fout)
        # json.dump(r_contextual.fs, fout)
    print ('=============', time.time() - t1)

print(time.time() - t1)
