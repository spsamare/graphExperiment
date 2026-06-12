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
from tabulate import tabulate

MOTIF_COUNTS = [1, 2, 4, 10, 31, 143]
SEED = 5


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
    f_data = os.path.isfile(file_data)
    f_motifs = os.path.isfile('motifs_' + file_data)
    f_org = os.path.isfile('org_' + file_data)

    if not (f_data and f_motifs and f_org):
        print('A file is missing...')
        print(file_data + str(f_data))
        print('motifs_' + file_data + str(f_data))
        print('org_' + file_data + str(f_data))
        sys.exit()
    else:
        dist_val = np.load(file_data)
        motif_dist = np.load('motifs_' + file_data)
        motif_distribution_detailed = np.load('org_' + file_data)

    # dist_val :: ndarray(num_attacks, all_configurations)
    # - dist_val[a, c] :: After 'a' attacks, the Wasserstien distance between initial and current
    #                     motif distributions for the configuration 'c'

    np.random.seed(SEED)  # reset seed for tractability

    # TEST
    val =[]
    best_configs = []
    for attk in range(1,num_max_attacks):
        attack_probabilities_init = np.zeros(num_max_attacks)
        s_vals = get_s_distribution(attk)  # (5 * num_max_attacks // 8)
        attack_probabilities_init[:s_vals.shape[0]] = s_vals
        exp_val_init = attack_probabilities_init @ dist_val
        best_config = np.argmin(exp_val_init)
        if not best_config in best_configs:
            best_configs.append(best_config)
        # print('Attacks: ', attk, '; Configuration: ', best_config)

    print('Number of best configurations: ', len(best_configs))

    best_configs_sorted = np.sort(best_configs)
    for attk in range(1,num_max_attacks):
        attack_probabilities_init = np.zeros(num_max_attacks)
        s_vals = get_s_distribution(attk)  # (5 * num_max_attacks // 8)
        attack_probabilities_init[:s_vals.shape[0]] = s_vals
        exp_val_init = attack_probabilities_init @ dist_val
        this_line = exp_val_init[best_configs_sorted].tolist()
        this_line.insert(0, attk)
        val.append(this_line)
        print(this_line)

    print(tabulate(val))


