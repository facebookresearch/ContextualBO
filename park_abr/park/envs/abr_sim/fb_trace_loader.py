# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# A modification of the adaptive video streaming environment in https://github.com/park-project/park

import json

import park


def get_chunk_time(trace, t_idx):
    if t_idx == len(trace[0]) - 1:
        return 1  # bandwidth last for 1 second
    else:
        return trace[0][t_idx + 1] - trace[0][t_idx]


def load_chunk_sizes():
    """
    chunk_sizes is a dict that with keys being context name and values
    being bytes of video chunk file at different bitrates for correpsonding
    traces.
    """
    # download video size folder if not existed
    video_folder = park.__path__[0] + "/envs/abr_sim/videos/"

    with open(video_folder + "fb_chunks_data_all.json", "r") as fp:
        chunk_sizes = json.load(fp)

    return chunk_sizes


def load_traces():
    """
        all_traces is a dict that with keys being context name and values
        being a dictionary with keys being trace uuid and values being bandwidths
            {
                "context name": {
                    "trace_id": (
                        [0.01, 1.0], # segment time (seconds)
                        [1e6, 2e6], # bandwidth (bytes per second)
                }
            }
    """
    # download video size folder if not existed
    trace_folder = park.__path__[0] + "/envs/abr_sim/fb_traces/"

    with open(trace_folder + "fb_traces_data_all.json", "r") as fp:
        all_traces = json.load(fp)

    return all_traces


def sample_trace(all_traces, irun):
    # deterministic
    trace_list = list(all_traces.keys())
    trace_list.sort()
    return trace_list[irun]


def random_sample_trace(all_traces, np_random):
    # weighted random sample based on trace length
    trace_list = list(all_traces.keys())
    trace_list.sort()

    all_p = [len(all_traces[trace_name][1]) for trace_name in trace_list]
    sum_p = float(sum(all_p))
    all_p = [p / sum_p for p in all_p]
    # sample a trace
    trace_idx = np_random.choice(len(trace_list), p=all_p)
    trace_uuid = trace_list[trace_idx]
    # sample a starting point
    init_t_idx = np_random.choice(len(all_traces[trace_uuid][0]))
    # return a trace and the starting t
    return trace_uuid, init_t_idx
