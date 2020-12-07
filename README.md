# ContextualBO
Code associated with paper ["High-Dimensional Contextual Policy Search with Unknown Context Rewards using Bayesian Optimization"](https://ai.facebook.com/research/publications/high-dimensional-contextual-policy-search-with-unknown-context-rewards-using-bayesian-optimization)

## Installation
To install the code clone the repo and install the dependencies as

    git clone https://github.com/facebookresearch/ContextualBO.git
    cd ContextualBO
    python3 -m pip install -r requirements.txt


Some of the baselines require additional packages that can not be pip-installed.

## Reproducing the experiments
This repository contains the code required to run the numerical experiments and the contextual Adaptive Bitrate (ABR) video playback experiment in the paper.

### Running Synethetic Benchmarks
The `benchmarks/` directory contains code for running the numerical experiments described in the paper. The benchmark problems are defined in `synethetic_problems.py`.

### Running Park ABR experiments
The `park_abr/` directory contains code for running the benchmark BO experiments described in the paper. The park problem is defined in `fb_abr_problem.py` and the simulator `park_abr/park/` is a folk of the adaptive video streaming environment in https://github.com/park-project/park. Each method has its own script for evaluating that method on the appropriate set of benchmark problems: `run_park_{method}.py`, where `{method}` is:

* `lcea`, for our method LCE-A, implemented in Ax
* `sac`, for our method SAC, implemented in Ax
* `standard_bo`, for Standard BO, implemented in Ax
* `alebo`, for ALEBO implemented in Ax
* `hesbo`, for HesBO implemented in Ax
* `rembo`, for REMBO implemented in Ax
* `addgpucb` for Add-GP-UCB via Dragonfly
* `cma_es` for CMA-ES
* `ebo` for Ensemble Bayesian Optimization
* `turbo` for TuRBO
* `non_contextual`, for Standard BO used for non-contextual optmization, implemented in Ax

See the paper for references for each of these methods. Each file explains what needs to be done in order to run the experiments for that method. For instance, `run_park_cma_es.py` requires installing `cma` from pip; `run_park_ebo.py` requires cloning a repository. See each file for its instructions.

### The contextual BO models and generation code
The actual implementation of the LCE-A, SAC, and LCE-M models is at: https://github.com/facebook/Ax/tree/master/ax/models/torch and https://github.com/pytorch/botorch/tree/master/botorch/models/


## License
This code is MIT Licensed, as found in the LICENSE file.
