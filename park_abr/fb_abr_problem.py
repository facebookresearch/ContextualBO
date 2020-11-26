#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import numpy as np
import park
from ax.utils.common.logger import get_logger


NUM_RUNS = 400
TH_DEFAULT = 3
TH_START_DEFAULT = 1


logger = get_logger(name="ABR_SIM_FB")


class Agent(object):
    def __init__(
        self,
        bw,
        bf,
        c,
        exp_weight,
        th=TH_DEFAULT,
        th_start=TH_START_DEFAULT,
        num_encodes=5,
    ):
        """Constructor.

        Args:
            bw: bandwidth prediction scaling term
            bf: buffer scaling term
            c: constant shift
            th: distance between thresholds for different
                bitrate
            th: starting threshold of bitrates
            exp_weight: expoential weights for bandwidth estimate
            num_encodes: number of encoding levels (available
                bitrates)
        """
        self.bw = bw
        self.bf = bf
        self.c = c
        self.num_encodes = num_encodes
        self.exp_weight = exp_weight
        self.th_levels = [th_start + i * th for i in range(num_encodes)]
        self.th_levels.append(np.inf)  # avoid empty sequence at loopup
        self.reset()

    def reset(self):
        self.prev_bw = []
        self.prev_t = []

    def exp_avg_bw(self, prev_bw, prev_t):
        """Expoential average bandwidth based on previous observations.

        Args:
            prev_bw: list of previous bandwidth observation
            prev_t: time intervals to the bandwidth observations
        """
        assert len(prev_bw) == len(prev_t)
        if len(prev_bw) == 0:
            return 0  # no previous observations

        prev_bw = np.array(prev_bw)
        prev_t = np.array(prev_t)
        prev_t_cumsum = np.cumsum(prev_t[::-1])[::-1]

        prev_t_exp = np.exp(-self.exp_weight * prev_t_cumsum)
        bw = np.sum(prev_bw * prev_t_exp) / np.sum(prev_t_exp)

        return bw

    def get_action(self, obs):
        # network bandwidth measurement for downloading the
        # last video chunk (with some normalization)
        curr_bw = obs[0] / 100000
        curr_t = obs[1]
        self.prev_bw.append(curr_bw)
        self.prev_t.append(curr_t)

        # estimate bandwidth with expoential average over past observations
        bw_est = self.exp_avg_bw(self.prev_bw, self.prev_t)

        # current video buffer occupancy with some normalization (see
        # https://github.com/park-project/park/blob/master/park/envs/abr_sim/abr.py
        # L82-L88 for more details)
        curr_bf = obs[2] / 10

        # here we assume the network bandwidth for downloading
        # the next chunk is the same (you can use more sophisticated method)
        th = self.bw * bw_est + self.bf * curr_bf + self.c

        # check which bitrate level is just below the threshold
        act = min(i for i in range(self.num_encodes + 1) if self.th_levels[i] > th)

        return act


class ContextualAgent(Agent):
    def __init__(self, bw_dict, bf_dict, c_dict, exp_weight_dict, num_encodes=5):
        """Contextual agent Constructor that resets bandwidths, buffer etc for
        different contexts.
        """
        self.bw_dict = bw_dict
        self.bf_dict = bf_dict
        self.c_dict = c_dict
        self.exp_weight_dict = exp_weight_dict
        self.num_encodes = num_encodes
        self.reset(context_name=None)

    def reset(self, context_name):
        self.prev_bw = []
        self.prev_t = []
        if context_name is not None:
            self.bw = self.bw_dict[context_name]
            self.bf = self.bf_dict[context_name]
            self.c = self.c_dict[context_name]
            self.th = TH_DEFAULT
            self.th_start = TH_START_DEFAULT
            self.th_levels = [
                self.th_start + i * self.th for i in range(self.num_encodes)
            ]
            self.th_levels.append(np.inf)  # avoid empty sequence at loopup
            self.exp_weight = self.exp_weight_dict[context_name]


class ParkNoncontextualRunner:
    def __init__(self, context_dict, max_eval=1000, return_context_reward=True):
        # For tracking iterations
        self.fs = []
        self.context_fs = []
        self.n_eval = 0
        self.max_eval = max_eval
        self.context_dict = context_dict
        self.return_context_reward = return_context_reward
        # define search space for non-dp setting
        self._base_parameters = [
            {
                "name": "bw",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float",
                "log_scale": False,
            },
            {
                "name": "bf",
                "type": "range",
                "bounds": [0.0, 3.0],
                "value_type": "float",
                "log_scale": False,
            },
            {
                "name": "c",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float",
                "log_scale": False,
            },
            {
                "name": "exp_weight",
                "type": "range",
                "bounds": [0.0001, 0.25],
                "value_type": "float",
                "log_scale": False,
            },
        ]
        self.n_params = len(self._base_parameters)

    @property
    def base_parameters(self) -> List[Dict]:
        return self._base_parameters

    def f(self, x):
        """
        x = [bw, bf, c, exp_weight]
        """
        if self.n_eval >= self.max_eval:
            raise StopIteration("Evaluation budget exhuasted")
        agent = Agent(bw=x[0], bf=x[1], c=x[2], exp_weight=x[3])
        rewards, context_rewards = run_non_contextual_experiments_multiple_times(
            agent=agent, context_dict=self.context_dict, num_runs=NUM_RUNS
        )  # Change this to 1 to make it faster
        f_x = np.mean(rewards)
        self.n_eval += 1
        self.fs.append(f_x)
        self.context_fs.append(context_rewards)
        if self.return_context_reward is False:
            return -f_x
        return -f_x, context_rewards  # because maximization


class ParkContextualRunner(ParkNoncontextualRunner):
    def __init__(
        self, num_contexts, context_dict, max_eval=1000, return_context_reward=True
    ):
        super().__init__(
            context_dict=context_dict,
            max_eval=max_eval,
            return_context_reward=return_context_reward,
        )
        self.num_contexts = num_contexts
        self.context_name_list = list(context_dict.keys())
        self._contextual_parameters = []
        for context_name in self.context_name_list:
            self._contextual_parameters.extend(
                [
                    {
                        "name": f"{self._base_parameters[j]['name']}_{context_name}",
                        "type": self._base_parameters[j]["type"],
                        "bounds": self._base_parameters[j]["bounds"],
                        "value_type": self._base_parameters[j]["value_type"],
                        "log_scale": self._base_parameters[j]["log_scale"],
                    }
                    for j in range(self.n_params)
                ]
            )

        self._decomposition = {
            f"{context_name}": [
                f"{self._base_parameters[j]['name']}_{context_name}"
                for j in range(self.n_params)
            ]
            for context_name in self.context_name_list
        }

    @property
    def contextual_parameters(self) -> List[Dict]:
        return self._contextual_parameters

    @property
    def contextual_parameter_decomposition(self) -> List[Dict]:
        return self._decomposition

    def f(self, x):
        """
        x = [bw_1, bf_1, c_1, exp_weight_1, bw_2, bf_2, c_2, exp_weight_2, ...]
        """
        if self.n_eval >= self.max_eval:
            raise StopIteration("Evaluation budget exhuasted")
        bw_dict = {
            f"{self.context_name_list[i]}": x[i * self.n_params]
            for i in range(self.num_contexts)
        }
        bf_dict = {
            f"{self.context_name_list[i]}": x[i * self.n_params + 1]
            for i in range(self.num_contexts)
        }
        c_dict = {
            f"{self.context_name_list[i]}": x[i * self.n_params + 2]
            for i in range(self.num_contexts)
        }
        exp_weight_dict = {
            f"{self.context_name_list[i]}": x[i * self.n_params + 3]
            for i in range(self.num_contexts)
        }
        agent = ContextualAgent(
            bw_dict=bw_dict,
            bf_dict=bf_dict,
            c_dict=c_dict,
            exp_weight_dict=exp_weight_dict,
        )
        # Change this to 1 to make it run faster
        rewards, context_rewards = run_contextual_experiments_multiple_times(
            agent=agent, context_dict=self.context_dict, num_runs=NUM_RUNS
        )
        f_x = np.mean(rewards)
        self.n_eval += 1
        self.fs.append(f_x)
        self.context_fs.append(context_rewards)
        if self.return_context_reward is False:
            return -f_x
        return -f_x, context_rewards


def run_contextual_experiments_multiple_times(agent, context_dict, num_runs):
    total_rewards = []
    context_rewards = {}
    for context_name, context_val in context_dict.items():
        env = park.make("abr_sim_fb")
        reward_list = []
        for irun in range(num_runs):
            obs = env.reset(context_val, irun)
            if len(obs) == 0:
                break
            agent.reset(context_name)
            done = False
            rewards = 0
            while not done:
                act = agent.get_action(obs)
                obs, reward, done, info = env.step(act)
                rewards += reward  # context weight could be applied here
            total_rewards.append(rewards)
            reward_list.append(rewards)
        context_rewards[context_name] = -(np.mean(reward_list))
    return total_rewards, context_rewards


def run_non_contextual_experiments_multiple_times(agent, context_dict, num_runs):
    total_rewards = []
    context_rewards = {}
    for context_name, context_val in context_dict.items():
        env = park.make("abr_sim_fb")
        reward_list = []
        for irun in range(num_runs):
            obs = env.reset(context_val, irun)
            if len(obs) == 0:
                break
            agent.reset()
            done = False
            rewards = 0
            while not done:
                act = agent.get_action(obs)
                obs, reward, done, info = env.step(act)
                rewards += reward  # context weight could be applied here
            total_rewards.append(rewards)
            reward_list.append(rewards)
        context_rewards[context_name] = -(np.mean(reward_list))
    return total_rewards, context_rewards
