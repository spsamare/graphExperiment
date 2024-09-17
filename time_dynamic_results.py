import sys

import numpy as np
from itertools import combinations, groupby
import networkx as nx
from matplotlib import pyplot as plt
from RSG_Graph_Motif_Counter.calculate_motif_distribution import calc_motif_distribution, render_motif_hist
from RSG_Graph_Motif_Counter.load_graph_motifs import load_motif_list
from tqdm import tqdm
from scipy.stats import wasserstein_distance as w_d
import os.path

MOTIF_COUNTS = [1, 2, 4, 10, 31, 143]
SEED = 0


def get_s_distribution(num_points, shift=0):
    x_range = 2
    if shift == -2:
        x_vals = np.linspace(0, x_range, num_points)
    elif shift == -1:
        x_vals = np.linspace(-x_range/2, x_range, num_points)
    elif shift == 2:
        x_vals = np.linspace(-x_range, 0, num_points)
    elif shift == 1:
        x_vals = np.linspace(-x_range, x_range/2, num_points)
    else:
        x_vals = np.linspace(-x_range, x_range, num_points)
    y_vals = []
    for x in x_vals:
        y_vals.append(s_curve(x))
    return np.array(y_vals) / np.sum(y_vals)


def s_curve(x):
    x_mod = np.abs(x) - 1
    y = 1 / (1 + np.exp(x_mod))
    return y


def get_moving_average(x, window_size=100):
    len_x = len(x)
    x_vals = []
    for i in range(len_x-window_size):
        x_vals.append(np.mean(x[i:(i+window_size)]))
    return x_vals


def get_block_average(x, window_size=100):
    len_x = len(x)
    x_vals = []
    for i in range(len_x//window_size):
        x_vals.append(np.mean(x[i*window_size:(i+1)*window_size]))
    return x_vals


if __name__ == "__main__":
    np.random.seed(SEED)  # fix seed
    # Graph creation
    num_initial_nodes = 50  # 50
    num_new_connections = 10  # 10
    num_max_attacks = 20  # 20
    #
    file_data = f'Data{SEED:02d}_{num_initial_nodes:03d}_{num_new_connections:02d}_{num_max_attacks:02d}.npy'

    if os.path.isfile(file_data) is False:
        print('Missing file')
        sys.exit()
    else:
        dist_val = np.load(file_data)

    # dist_val :: ndarray(num_attacks, all_configurations)
    # - dist_val[a, c] :: After 'a' attacks, the Wasserstien distance between initial and current
    #                     motif distributions for the configuration 'c'

    np.random.seed(SEED)  # reset seed for tractability

    attack_probabilities_init = np.zeros(num_max_attacks)
    s_vals = get_s_distribution(15)  # (5 * num_max_attacks // 8)
    attack_probabilities_init[:s_vals.shape[0]] = s_vals
    exp_val_init = attack_probabilities_init @ dist_val
    #
    attack_probabilities_now = np.zeros(num_max_attacks)
    s_vals = get_s_distribution(11)  # (num_max_attacks // 4)
    attack_probabilities_now[:s_vals.shape[0]] = s_vals
    exp_val_now = attack_probabilities_now @ dist_val

    # temporal settings
    original_duration = 10000 * 2
    transition_duration = 5000 * 2
    new_duration = 2*original_duration - transition_duration

    #
    scale_val = 1000
    robust_config = np.argmin(exp_val_init)
    resilient_config = np.argmin(exp_val_init)

    attack_list = []
    loss_robust = []
    loss_resilient = []

    for t in range(original_duration+transition_duration+new_duration):
        if t < original_duration:
            cur_attack = np.random.choice(num_max_attacks, 1, p=attack_probabilities_init)
        else:
            cur_attack = np.random.choice(num_max_attacks, 1, p=attack_probabilities_now)
            if t == original_duration+transition_duration:
                resilient_config = np.argmin(exp_val_now)

        loss_robust.append(scale_val * dist_val[cur_attack, robust_config][0])
        loss_resilient.append(scale_val * dist_val[cur_attack, resilient_config][0])
        attack_list.append(cur_attack[0])

    # plotting
    w_size = 100
    avg_loss_robust = get_block_average(loss_robust, window_size=w_size)
    avg_loss_resilient = get_block_average(loss_resilient, window_size=w_size)

    w_size = 50  # 2500 * 2
    #avg_loss_robust = get_moving_average(loss_robust, window_size=w_size)
    #avg_loss_resilient = get_moving_average(loss_resilient, window_size=w_size)
    avg_loss_robust = get_moving_average(avg_loss_robust, window_size=w_size)
    avg_loss_resilient = get_moving_average(avg_loss_resilient, window_size=w_size)

    T = 500
    plt.plot(avg_loss_robust[0:T], label='Robust')
    plt.plot(avg_loss_resilient[0:T], label='Resilient')
    #
    plt.xlabel("Time")
    plt.ylabel("Average Loss")
    leg = plt.legend(loc='upper center')
    # plt.xlim((0, ind[-1]))
    # plt.grid(linestyle=':')
    #
    plt.show()
