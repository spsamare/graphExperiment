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


def draw_graph(num_nodes, adj_matrix, con_matrix):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] == 1:
                graph.add_edge(i, j, style='-')
    #
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            if con_matrix[i, j] == 1:
                graph.add_edge(i, j, style='--')
    #
    # pos = nx.circular_layout(graph)
    edges = graph.edges()
    style = [graph[u][v]['style'] for u, v in edges]
    nx.draw(graph, edgelist=edges, style=style, node_size=20)
    plt.show()


def generate_random_connected_graph(num_nodes=3, probability=None):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    """
    if probability is None:
        probability = min(2/(num_nodes - 1), 1)

    edges = combinations(range(num_nodes), 2)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge_id = np.random.choice(len(node_edges), 1)[0]
        random_edge = node_edges[random_edge_id]
        graph.add_edge(*random_edge)
        for e in node_edges:
            if np.random.random() < probability:
                graph.add_edge(*e)
    return graph


def generate_graph(num_nodes=3, num_connects=1, new_con_node=0):
    assert num_nodes > 1, "Number of nodes should be larger than one"

    graph_connected = generate_random_connected_graph(num_nodes=num_nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        neighbor_list = graph_connected.neighbors(i)
        for j in neighbor_list:
            adj_matrix[i, j] = 1

    node_deg = np.sum(adj_matrix, 0)
    # print(node_deg)
    node_index = np.argsort(node_deg)[::-1]
    # print(node_index)

    adj_matrix_new = np.zeros((num_nodes, num_nodes), dtype=int)
    graph = nx.Graph()
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            adj_matrix_new[i, j] = adj_matrix[node_index[i], node_index[j]]
            if adj_matrix_new[i, j] == 1:
                graph.add_edge(i, j)

    for i in range(num_nodes//2, num_nodes - 1):
        if np.sum(adj_matrix_new[i, num_nodes//2:]) < 1:  # add node to ensure connectivity
            j = np.random.randint(num_nodes//2, num_nodes)
            while j == i:
                j = np.random.randint(num_nodes // 2, num_nodes)
            adj_matrix_new[i, j] = 1
            graph.add_edge(i, j)

    adj_matrix_new = adj_matrix_new + np.transpose(adj_matrix_new)

    # compute feasible connections
    up_triangle = adj_matrix[np.triu_indices(num_nodes, 1)]
    temp_mat = np.ones((num_nodes, num_nodes), dtype=int)
    temp_mat[np.triu_indices(num_nodes, 1)] = up_triangle
    (no_row, no_col) = np.where(temp_mat == 0)
    start_index = np.argmax(no_row > new_con_node)  # new connections are only for the nodes above <new_con_node>
    no_list = np.arange(no_row.shape[0])
    allowed_connections = np.random.choice(no_list[start_index:], num_connects, False)
    con_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for k in allowed_connections:
        con_matrix[no_row[k], no_col[k]] = 1

    return adj_matrix_new, graph, con_matrix


def make_graph(num_nodes, adj_matrix):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] == 1:
                graph.add_edge(i, j)
    return graph


def generate_graph_old(num_nodes=3):
    assert num_nodes > 1, "Number of nodes should be larger than one"

    p = .01
    # up_triangle = np.random.randint(0, 2, num_nodes * (num_nodes - 1) // 2)
    up_triangle = np.random.binomial(1, p, num_nodes * (num_nodes - 1) // 2)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    adj_matrix[np.triu_indices(num_nodes, 1)] = up_triangle
    adj_matrix = adj_matrix + np.transpose(adj_matrix)

    node_deg = np.sum(adj_matrix, 0)
    # print(node_deg)
    node_index = np.argsort(node_deg)[::-1]
    # print(node_index)

    adj_matrix_new = np.zeros((num_nodes, num_nodes), dtype=int)
    graph = nx.Graph()
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            adj_matrix_new[i, j] = adj_matrix[node_index[i], node_index[j]]
            if adj_matrix_new[i, j] == 1:
                graph.add_edge(i, j)

    adj_matrix_new = adj_matrix_new + np.transpose(adj_matrix_new)

    return adj_matrix_new, graph


def get_reduced_adjacency(adj_matrix, con_matrix, num_nodes=1):
    assert adj_matrix.shape[0] - num_nodes > 1, "Reduced network should have at least two nodes"
    return adj_matrix[num_nodes:, num_nodes:], con_matrix[num_nodes:, num_nodes:]


def get_possible_configurations(adj_matrix, con_matrix, max_attacks=1):
    num_nodes = adj_matrix.shape[0]

    combination_list = [adj_matrix.copy()]
    modification_indicator = [np.ones(max_attacks-1)]

    (no_row, no_col) = np.where(con_matrix == 1)
    no_list = np.arange(no_row.shape[0])

    print("Generating all possibilities")
    for r in tqdm(range(1, no_row.shape[0]+1)):
        for combination in combinations(no_list, r):
            x = adj_matrix.copy()
            # print(combination)
            node_id = 0
            for k in combination:
                x[no_row[k], no_col[k]] = 1
                x[no_col[k], no_row[k]] = 1
                node_id = max(node_id, min(no_row[k], no_col[k]))
            combination_list.append(x)
            indict = np.zeros(max_attacks-1)
            # node_id = max(np.minimum(no_row[combination], no_col[combination]))
            indict[:node_id] = 1
            modification_indicator.append(indict)

    return combination_list, modification_indicator


def get_possible_configurations_old(adj_matrix, con_matrix=None, max_motif=0, min_motif=1):
    num_nodes = adj_matrix.shape[0]
    temp_mat = np.ones((num_nodes, num_nodes), dtype=int)
    combination_list = []

    if max_motif > 0:
        graph = nx.Graph()
        graph.add_nodes_from(*[range(num_nodes)])
        for i in range(0, num_nodes-1):
            for j in range(i+1, num_nodes):
                if adj_matrix[i, j] == 1:
                    graph.add_edge(i, j)
        nx.draw(graph)
        plt.show()
        motifs = load_motif_list([*range(min_motif, max_motif + 1)])
        distribution_list = []
        distribution_list_detailed = []

    if con_matrix is None:
        up_triangle = adj_matrix[np.triu_indices(num_nodes, 1)]
        temp_mat[np.triu_indices(num_nodes, 1)] = up_triangle
        (no_row, no_col) = np.where(temp_mat == 0)
        no_list = np.arange(no_row.shape[0])

        for r in tqdm(range(0, no_row.shape[0] + 1)):
            for combination in combinations(no_list, r):
                x = adj_matrix.copy()
                if max_motif > 0:
                    graph_mod =graph.copy()
                for k in combination:
                    x[no_row[k], no_col[k]] = 1
                    x[no_col[k], no_row[k]] = 1
                    if max_motif > 0:
                        graph_mod.add_edge(no_row[k], no_col[k])
                        motif_d_detailed, motif_d = compute_motifs(
                            graph_mod,
                            motifs=motifs,
                            max_motif=max_motif,
                            min_motif=min_motif
                        )
                        distribution_list.append(motif_d)
                        distribution_list_detailed.append(motif_d_detailed)
                combination_list.append(x)
    else:
        (no_row, no_col) = np.where(con_matrix == 1)
        no_list = np.arange(no_row.shape[0])

        for r in tqdm(range(no_row.shape[0]+1)):
            for combination in combinations(no_list, r):
                x = adj_matrix.copy()
                if max_motif > 0:
                    graph_mod =graph.copy()
                for k in combination:
                    x[no_row[k], no_col[k]] = 1
                    x[no_col[k], no_row[k]] = 1
                    if max_motif > 0:
                        graph_mod.add_edge(no_row[k], no_col[k])
                if max_motif > 0:
                    # nx.draw(graph_mod)
                    # plt.show()
                    motif_d_detailed, motif_d = compute_motifs(
                        graph_mod,
                        motifs=motifs,
                        max_motif=max_motif,
                        min_motif=min_motif
                    )
                    distribution_list.append(motif_d)
                    distribution_list_detailed.append(motif_d_detailed)
                combination_list.append(x)

    if max_motif > 0:
        return combination_list, distribution_list_detailed, distribution_list
    else:
        return combination_list


def compute_motifs(graph, motifs, max_motif, min_motif=1):
    hist = calc_motif_distribution(graph, motifs)
    hist = hist / sum(hist)
    # print(hist)

    hist_mod = []
    i_val = 0
    min_count = 0 if min_motif <= 1 else MOTIF_COUNTS[min_motif-2]
    for _ in range(min_motif - 1, max_motif):
        hist_mod.append(sum(hist[i_val:(MOTIF_COUNTS[_]-min_count)]))
        i_val = _

    return hist, hist_mod


def motif_search(num_nodes, adj_matrix_list, compute_indicate_list, max_attacks, motifs, max_motif, min_motif=1):
    # For first attack
    d_list = []
    d_det_list = []
    print("Motif computation for attack 1")
    for k in tqdm(range(len(adj_matrix_list))):
        graph = make_graph(num_nodes=num_nodes, adj_matrix=adj_matrix_list[k])
        distribution_detailed, distribution = compute_motifs(
            graph=graph, motifs=motifs, max_motif=max_motif, min_motif=min_motif
        )
        d_list.append(distribution)
        d_det_list.append(distribution_detailed)

    dist_list = [d_list]
    dist_det_list = [d_det_list]

    # For the remaining attacks
    for attack in range(1, max_attacks):
        print("Motif computation for attack", attack + 1)
        d_list = []
        d_det_list = []
        for k in tqdm(range(len(adj_matrix_list))):
            if compute_indicate_list[k][attack-1] == 0:
                distribution = d_list[0]
                distribution_detailed = d_det_list[0]
            else:
                graph = make_graph(num_nodes=num_nodes-attack, adj_matrix=adj_matrix_list[k][attack:, attack:])
                distribution_detailed, distribution = compute_motifs(
                    graph=graph, motifs=motifs, max_motif=max_motif, min_motif=min_motif
                )
            d_list.append(distribution)
            d_det_list.append(distribution_detailed)
        dist_list.append(d_list)
        dist_det_list.append(d_det_list)

    return dist_det_list, dist_list


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


if __name__ == "__main__":
    np.random.seed(SEED)  # fix seed
    # Graph creation
    num_initial_nodes = 50  # 50
    num_new_connections = 10  # 10
    num_max_attacks = 20  # 20
    #
    # Drawing only
    adj_mat_init, G_init, con_mat = generate_graph(
        num_nodes=num_initial_nodes,
        num_connects=num_new_connections,
        new_con_node=num_max_attacks // 2
    )
    draw_graph(num_nodes=num_initial_nodes, adj_matrix=adj_mat_init, con_matrix=con_mat)
    #
    file_data = f'Data{SEED:02d}_{num_initial_nodes:03d}_{num_new_connections:02d}_{num_max_attacks:02d}.npy'

    if os.path.isfile(file_data) is False:
        adj_mat_init, G_init, con_mat = generate_graph(
            num_nodes=num_initial_nodes,
            num_connects=num_new_connections,
            new_con_node=num_max_attacks // 2
        )
        # nx.draw(G_init)
        # plt.show()

        # Compute motif distribution
        max_motifs = 4
        min_motifs = max(max_motifs - 0, 1)
        motif_list = load_motif_list([*range(min_motifs, max_motifs + 1)])  # load_motif_list([1, 2, 3, 4])
        motif_distribution_detailed, motif_distribution = compute_motifs(
            G_init, motifs=motif_list, max_motif=max_motifs, min_motif=min_motifs
        )
        # render_motif_hist(motif_distribution_detailed, motif_list)
        # render_motif_hist(motif_distribution, motif_list)

        """"""
        # First attack
        adj_mat_mod, con_mat_mod = get_reduced_adjacency(
            adj_matrix=adj_mat_init, con_matrix=con_mat, num_nodes=1
        )
        # print(adj_mat_mod)
        # Mimicking rest of the attacks
        config_list, indict_list = get_possible_configurations(
            adj_matrix=adj_mat_mod,
            con_matrix=con_mat_mod, max_attacks=num_max_attacks
        )
        print('Num configurations', len(config_list))

        # Compute motif distributions for all possible configurations
        distribution_list_detailed, distribution_list = motif_search(
            num_nodes=num_initial_nodes - 1, adj_matrix_list=config_list,
            compute_indicate_list=indict_list, max_attacks=num_max_attacks,
            motifs=motif_list, max_motif=max_motifs, min_motif=min_motifs
        )

        # Calculate Wasserstein distances & visualizing
        dist_val = []
        for atk in range(num_max_attacks):
            # dist = []
            # for d in distribution_list[atk]:
            #     dist.append(w_d(d, motif_distribution))
            dist_det = []
            for d in distribution_list_detailed[atk]:
                dist_det.append(w_d(d, motif_distribution_detailed))
            dist_val.append(dist_det)
            plt.plot(dist_det, label=str(atk + 1))
            # plt.plot(dist_det)

        dist_val = np.array(dist_val)
        attack_probabilities = np.random.random(num_max_attacks)
        attack_probabilities = np.array(attack_probabilities / np.sum(attack_probabilities))
        exp_val = attack_probabilities @ dist_val
        plt.plot(exp_val, label='Known stressor')
        best_setting = np.argmin(exp_val)

        new_probabilities = np.random.random(num_max_attacks)
        new_probabilities[num_max_attacks // 2:] = 0
        new_probabilities = np.array(new_probabilities / np.sum(new_probabilities))
        new_val = new_probabilities @ dist_val
        plt.plot(new_val, label='Unknown stressor')
        best_setting_new = np.argmin(new_val)

        plt.plot(
            [best_setting, best_setting, best_setting_new],
            [exp_val[best_setting], new_val[best_setting], new_val[best_setting_new]],
            ls="", marker="o"
        )

        leg = plt.legend(loc='upper center')
        plt.show()

        # save data
        np.save(file_data, dist_val)
    else:
        dist_val = np.load(file_data)

    # dist_val :: ndarray(num_attacks, all_configurations)
    # - dist_val[a, c] :: After 'a' attacks, the Wasserstien distance between initial and current
    #                     motif distributions for the configuration 'c'

    # Modeling stressor and its drifts, and then comparing robust & resilient solutions
    np.random.seed(SEED)  # reset seed for tractability

    # Option 1: UNIFORM attack probability
    """
    #
    stress_mu_init = num_max_attacks / 2  # define initial mean
    stress_max_init = 2 * num_max_attacks // 3  # define initial expected max attacks
    #
    attack_probabilities_init = np.random.rand(num_max_attacks)  # np.random.poisson(stress_mu_init, num_max_attacks)
    attack_probabilities_init[stress_max_init:] = 0
    attack_probabilities_init = np.array(attack_probabilities_init / np.sum(attack_probabilities_init))
    exp_val_init = attack_probabilities_init @ dist_val

    # Configurations
    robust_config = np.argmin(exp_val_init)

    # Results
    stress_model = [attack_probabilities_init]
    resilient_configs = [np.argmin(exp_val_init)]
    loss_robust = [exp_val_init[robust_config]]
    loss_resilient = [exp_val_init[resilient_configs[0]]]

    # Change stressor
    num_steps = 11
    for step in range(num_steps-1):
        # modify attack probabilities
        stress_mu = (num_max_attacks-1) * np.random.rand() + 1
        stress_max = np.random.randint(num_max_attacks//4, num_max_attacks+1)
        # print('mu:', stress_mu, ', max:', stress_max)
        #
        attack_probabilities_now = np.random.rand(num_max_attacks)  # np.random.poisson(stress_mu, num_max_attacks)
        attack_probabilities_now[stress_max:] = 0
        attack_probabilities_now = np.array(attack_probabilities_now / np.sum(attack_probabilities_now))
        stress_model.append(attack_probabilities_now)
        #
        exp_val_now = attack_probabilities_now @ dist_val
        #
        resilient_configs.append(np.argmin(exp_val_now))
        #
        loss_robust.append(exp_val_now[robust_config])
        loss_resilient.append(exp_val_now[resilient_configs[-1]])

    # Plotting
    ind = np.arange(num_steps)
    plt.step(ind, loss_robust, where='post', label='Robust')
    plt.step(ind, loss_resilient, where='post', label='Resilient')
    #
    plt.xlabel("Intervals")
    plt.ylabel("Loss")
    leg = plt.legend(loc='upper center')
    #
    plt.show()
    """

    # Option 2: Monte Carlo - Uniform
    """
    probs = [attack_probabilities_init]
    num_loops = 1000
    rob_val = 0.
    res_val = 0.
    scale_val = 1000
    prob_val = np.zeros(num_max_attacks)
    loss_robust = [exp_val_init[robust_config]]
    loss_resilient = [exp_val_init[robust_config]]
    for step in range(num_loops*(num_max_attacks-1)):
        # modify attack probabilities
        # stress_mu = (num_max_attacks - 1) * np.random.rand() + 1
        stress_max = step // num_loops
        # print('mu:', stress_mu, ', max:', stress_max)
        #
        attack_probabilities_now = np.random.rand(num_max_attacks)  # np.random.poisson(stress_mu, num_max_attacks)
        attack_probabilities_now[(stress_max+1):] = 0
        attack_probabilities_now = np.array(attack_probabilities_now / np.sum(attack_probabilities_now))
        stress_model.append(attack_probabilities_now)
        #
        exp_val_now = attack_probabilities_now @ dist_val
        #
        resilient_config = np.argmin(exp_val_now)
        #
        rob_val += exp_val_now[robust_config]
        res_val += exp_val_now[resilient_config]
        prob_val += attack_probabilities_now

        if (step + 1) % num_loops == 0:
            loss_resilient.append(res_val / num_loops)
            loss_robust.append(rob_val / num_loops)
            probs.append(prob_val / num_loops)
            prob_val = np.zeros(num_max_attacks)
            res_val = 0
            rob_val = 0

    # Plotting
    loss_resilient.append(loss_resilient[-1])
    loss_robust.append(loss_robust[-1])

    loss_robust = np.array(loss_robust)
    loss_resilient = np.array(loss_resilient)
    # print(probs)
    ind = np.arange(num_max_attacks + 1)
    plt.step(ind, scale_val * loss_robust, where='post', label='Robust')
    plt.step(ind, scale_val * loss_resilient, where='post', label='Resilient')
    #
    plt.xlabel("Intervals")
    plt.ylabel("Loss")
    leg = plt.legend(loc='upper center')
    plt.grid()
    #
    plt.show()

    for p in probs:
        plt.bar(np.arange(p.shape[0]) + 1, p)
        plt.xlabel('Num. attacks')
        plt.ylabel('Probability')
        plt.show()
    """

    # Option 3: S-Curve distribution
    """"""
    attack_probabilities_init = np.zeros(num_max_attacks)
    s_vals = get_s_distribution(5 * num_max_attacks // 8)
    attack_probabilities_init[:s_vals.shape[0]] = s_vals
    exp_val_init = attack_probabilities_init @ dist_val
    #
    scale_val = 1000
    loss_robust = np.zeros((5, num_max_attacks))
    loss_resilient = np.zeros((5, num_max_attacks))
    robust_config = np.argmin(exp_val_init)
    #
    probs = [attack_probabilities_init]
    loss_robust = [exp_val_init[robust_config]]
    loss_resilient = [exp_val_init[robust_config]]

    for k in range(num_max_attacks // 4 - 1, num_max_attacks):
        attack_probabilities_now = np.zeros(num_max_attacks)
        s_vals = get_s_distribution(k + 1)
        attack_probabilities_now[:s_vals.shape[0]] = s_vals
        #
        exp_val_now = attack_probabilities_now @ dist_val
        #
        loss_robust.append(exp_val_now[robust_config])
        loss_resilient.append(np.min(exp_val_now))
        probs.append(attack_probabilities_now)

    # Plotting
    loss_resilient.append(loss_resilient[-1])
    loss_robust.append(loss_robust[-1])

    loss_robust = np.array(loss_robust)
    loss_resilient = np.array(loss_resilient)
    # print(probs)
    ind = np.arange(num_max_attacks - (num_max_attacks // 4 - 1) + 2)
    plt.step(ind, scale_val * loss_robust, where='post', label='Robust')
    plt.step(ind, scale_val * loss_resilient, where='post', label='Resilient')
    #
    plt.xlabel("Intervals")
    plt.ylabel("Loss")
    leg = plt.legend(loc='upper center')
    plt.xlim((0, ind[-1]))
    plt.grid(linestyle=':')
    #
    plt.show()

    for p in probs:
        plt.bar(np.arange(p.shape[0]) + 1, p)
        plt.xlabel('Num. attacks')
        plt.ylabel('Probability')
        plt.xlim((0, num_max_attacks))
        plt.grid(linestyle=':')
        plt.show()
    """"""

    # Option 4: Matrix Plot with S curve
    attack_probabilities_init = np.zeros(num_max_attacks)
    s_vals = get_s_distribution(5 * num_max_attacks // 8)
    attack_probabilities_init[:s_vals.shape[0]] = s_vals
    exp_val_init = attack_probabilities_init @ dist_val
    #
    scale_val = 1000
    loss_robust = np.zeros((5, num_max_attacks))
    loss_resilient = np.zeros((5, num_max_attacks))
    robust_config = np.argmin(exp_val_init)

    for s in range(5):
        for k in range(num_max_attacks//4 - 1, num_max_attacks):
            attack_probabilities_now = np.zeros(num_max_attacks)
            s_vals = get_s_distribution(k + 1, s - 2)
            attack_probabilities_now[:s_vals.shape[0]] = s_vals
            #
            exp_val_now = attack_probabilities_now @ dist_val
            #
            loss_robust[s, k] = exp_val_now[robust_config]
            loss_resilient[s, k] = np.min(exp_val_now)

    loss_difference = loss_robust - loss_resilient
    plt.imshow(loss_difference[:, num_max_attacks//4 - 1:],
               cmap='viridis', interpolation='nearest')
    plt.show()

    """"""
    """
    dist_det = [w_d(motif_distribution_detailed, motif_distribution_detailed)]
    dist = [w_d(motif_distribution, motif_distribution)]
    for atk in range(num_max_attacks):
        adj_mat_mod, con_mat_mod = get_reduced_adjacency(
            adj_matrix=adj_mat_init, con_matrix=con_mat, num_nodes=atk+1
        )
        print('Attack:', atk+1, '-- New links:', np.sum(con_mat_mod))
        G_mod = make_graph(num_nodes=num_initial_nodes-atk-1, adj_matrix=adj_mat_mod)
        temp_det, temp_ = compute_motifs(
            G_mod, motifs=motif_list, max_motif=max_motifs, min_motif=min_motifs
        )
        dist_det.append(w_d(temp_det, motif_distribution_detailed))
        dist.append(w_d(temp_, motif_distribution))

    plt.plot(dist_det, label='Detailed')
    plt.plot(dist, label='Simplified')
    plt.xlabel("Num. attacks")
    plt.ylabel("Distance")
    leg = plt.legend(loc='upper center')
    plt.show()
    """
