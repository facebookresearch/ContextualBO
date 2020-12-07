# Folk of the adaptive video streaming environment in https://github.com/park-project/park

import os

import numpy as np
import park


def get_chunk_time(trace, t_idx):
    if t_idx == len(trace[0]) - 1:
        return 1  # bandwidth last for 1 second
    else:
        return trace[0][t_idx + 1] - trace[0][t_idx]


def load_chunk_sizes():
    # bytes of video chunk file at different bitrates

    # source video: "Envivio-Dash3" video H.264/MPEG-4 codec
    # at bitrates in {300,750,1200,1850,2850,4300} kbps

    # original video file:
    # https://github.com/hongzimao/pensieve/tree/master/video_server

    # download video size folder if not existed
    video_folder = park.__path__[0] + "/envs/abr_sim/videos/"

    chunk_sizes = np.load(video_folder + "video_sizes.npy")

    return chunk_sizes


def load_traces():
    # download video size folder if not existed
    trace_folder = park.__path__[0] + "/envs/abr_sim/traces/"

    all_traces = []

    for trace in os.listdir(trace_folder):

        all_t = []
        all_bandwidth = []

        with open(trace_folder + trace, "rb") as f:
            for line in f:
                parse = line.split()
                all_t.append(float(parse[0]))
                all_bandwidth.append(float(parse[1]))

        all_traces.append((all_t, all_bandwidth))

    return all_traces


def sample_trace(all_traces, np_random):
    # weighted random sample based on trace length
    all_p = [len(trace[1]) for trace in all_traces]
    sum_p = float(sum(all_p))
    all_p = [p / sum_p for p in all_p]
    # sample a trace
    trace_idx = np_random.choice(len(all_traces), p=all_p)
    # sample a starting point
    init_t_idx = np_random.choice(len(all_traces[trace_idx][0]))
    # return a trace and the starting t
    return all_traces[trace_idx], init_t_idx
