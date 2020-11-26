# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# A modification of the adaptive video streaming environment in https://github.com/park-project/park

from collections import deque

import numpy as np
from ax.utils.common.logger import get_logger
from park import logger
from park.envs.abr_sim.abr import ABRSimEnv
from park.envs.abr_sim.fb_trace_loader import (
    get_chunk_time,
    load_chunk_sizes,
    load_traces,
    sample_trace,
)


logger = get_logger(name="ABR_SIM_FB")


class ABRSimFBEnv(ABRSimEnv):
    """
    ABRSimEnv in FB setting.

    Adapt bitrate during a video playback with varying network conditions.
    The objective is to (1) reduce stall (2) increase video quality and
    (3) reduce switching between bitrate levels. Ideally, we would want to
    *simultaneously* optimize the objectives in all dimensions.

    * STATE *
        [The throughput estimation of the past chunk (chunk size / elapsed time),
        download time (i.e., elapsed time since last action), current buffer ahead,
        number of the chunks until the end, the bitrate choice for the past chunk,
        current chunk size of bitrate 1, chunk size of bitrate 2,
        ..., chunk size of bitrate 5]

        Note: we need the selected bitrate for the past chunk because reward has
        a term for bitrate change, a fully observable MDP needs the bitrate for past chunk

    * ACTIONS *
        Which bitrate to choose for the current chunk, represented as an integer in [0, 5]

    * REWARD *
        At current time t, the selected bitrate is b_t, the stall time between
        t to t + 1 is s_t, then the reward r_t is
        b_{t} - 2.8 * s_{t} - 0.5 * |b_t - b_{t-1}|

    * REFERENCE *
        Section 4.2, Section 5.1
        Neural Adaptive Video Streaming with Pensieve
        H Mao, R Netravali, M Alizadeh
        https://dl.acm.org/citation.cfm?id=3098843

        Figure 1b, Section 6.2 and Appendix J
        Variance Reduction for Reinforcement Learning in Input-Driven Environments.
        H Mao, SB Venkatakrishnan, M Schwarzkopf, M Alizadeh.
        https://openreview.net/forum?id=Hyg1G2AqtQ

        A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP
        X Yin, A Jindal, V Sekar, B Sinopoli
        https://dl.acm.org/citation.cfm?id=2787486
    """

    def __init__(self):
        # observation and action space
        self.setup_space()
        # set up seed
        self.seed(42)
        # load all trace files
        self.all_traces = load_traces()
        # load all video chunk sizes
        self.all_chunk_sizes = load_chunk_sizes()
        # mapping between action and bitrate level
        self.bitrate_map = [0.3, 0.75, 1.2, 1.85, 2.85, 4.3]  # Mbps
        # how many past throughput to report
        self.past_chunk_len = 8

    def reset(self, context_setup, irun):
        # context_setup = {"name": "context_0", "delay": delay}
        context_name = context_setup["name"]

        # load trace
        if irun >= len(self.all_traces[context_name]):
            return []
        trace_uuid = sample_trace(self.all_traces[context_name], irun)
        self.trace = self.all_traces[context_name][trace_uuid]
        self.curr_t_idx = 0

        # load chunk
        self.chunk_sizes = self.all_chunk_sizes[context_name][trace_uuid]  # sample
        self.chunk_idx = 0
        self.total_num_chunks = len(self.chunk_sizes[0])
        # assert number of chunks for different bitrates are all the same
        assert len(np.unique([len(chunk_size) for chunk_size in self.chunk_sizes])) == 1

        self.delay = context_setup.get("delay", 0.0)
        self.chunk_time_left = get_chunk_time(self.trace, self.curr_t_idx)

        self.buffer_size = 0.0  # initial download time not counted
        self.past_action = None
        self.past_chunk_throughputs = deque(maxlen=self.past_chunk_len)
        self.past_chunk_download_times = deque(maxlen=self.past_chunk_len)
        for _ in range(self.past_chunk_len):
            self.past_chunk_throughputs.append(0)
            self.past_chunk_download_times.append(0)
        return self.observe()

    def step(self, action):

        # 0 <= action < num_servers
        assert self.action_space.contains(action)

        # Note: sizes are in bytes, times are in seconds
        chunk_size = self.chunk_sizes[action][self.chunk_idx]

        # compute chunk download time based on trace
        delay = self.delay  # in seconds

        # keep experiencing the network trace
        # until the chunk is downloaded
        while chunk_size > 1e-8:  # floating number business

            throuput = self.trace[1][self.curr_t_idx]  # bytes/second
            throuput = throuput / 3.0
            throuput = max(throuput, 0)

            chunk_time_used = min(self.chunk_time_left, chunk_size / throuput)

            chunk_size -= throuput * chunk_time_used
            self.chunk_time_left -= chunk_time_used
            delay += chunk_time_used

            if self.chunk_time_left == 0:

                self.curr_t_idx += 1
                if self.curr_t_idx == len(self.trace[1]):
                    self.curr_t_idx = 0

                self.chunk_time_left = get_chunk_time(self.trace, self.curr_t_idx)

        # compute buffer size
        rebuffer_time = max(delay - self.buffer_size, 0)

        # update video buffer
        self.buffer_size = max(self.buffer_size - delay, 0)
        self.buffer_size += 4.0  # each chunk is 4 seconds of video

        # cap the buffer size
        self.buffer_size = min(self.buffer_size, 40.0)

        # bitrate change penalty
        if self.past_action is None:
            bitrate_change = 0
        else:
            bitrate_change = np.abs(
                self.bitrate_map[action] - self.bitrate_map[self.past_action]
            )

        # linear reward
        # (https://dl.acm.org/citation.cfm?id=3098843 section 5.1, QoE metrics (1))
        reward = self.bitrate_map[action] - 2.8 * rebuffer_time - 0.5 * bitrate_change

        # store action for future bitrate change penalty
        self.past_action = action

        # update observed network bandwidth and duration
        self.past_chunk_throughputs.append(
            self.chunk_sizes[action][self.chunk_idx] / float(delay)
        )
        self.past_chunk_download_times.append(delay)

        # advance video
        self.chunk_idx += 1
        done = self.chunk_idx == self.total_num_chunks

        return (
            self.observe(),
            reward,
            done,
            {
                "bitrate": self.bitrate_map[action],
                "stall_time": rebuffer_time,
                "bitrate_change": bitrate_change,
            },
        )
