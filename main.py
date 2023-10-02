import numpy as np
from itertools import combinations, groupby
import networkx as nx
from matplotlib import pyplot as plt
from RSG_Graph_Motif_Counter.calculate_motif_distribution import calc_motif_distribution, render_motif_hist
from RSG_Graph_Motif_Counter.load_graph_motifs import load_motif_list
from tqdm import tqdm
from scipy.stats import wasserstein_distance as w_d

MOTIF_COUNTS = [1, 2, 4, 10, 31, 143]


def generate_random_connected_graph(num_nodes=3, probability=None):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    """
    if probability is None:
        probability = 1/(num_nodes - 1)

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


def generate_graph(num_nodes=3, num_connects=1):
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

    adj_matrix_new = adj_matrix_new + np.transpose(adj_matrix_new)

    # compute feasible connections
    up_triangle = adj_matrix[np.triu_indices(num_nodes, 1)]
    temp_mat = np.ones((num_nodes, num_nodes), dtype=int)
    temp_mat[np.triu_indices(num_nodes, 1)] = up_triangle
    (no_row, no_col) = np.where(temp_mat == 0)
    start_index = np.argmax(no_row > 0)
    no_list = np.arange(no_row.shape[0])
    allowed_connections = np.random.choice(no_list[start_index:], num_connects, False)
    con_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for k in allowed_connections:
        con_matrix[no_row[k], no_col[k]] = 1

    return adj_matrix_new, graph, con_matrix


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


def get_possible_configurations(adj_matrix, con_matrix=None, max_motif=0):
    num_nodes = adj_matrix.shape[0]
    temp_mat = np.ones((num_nodes, num_nodes), dtype=int)
    combination_list = [adj_matrix]

    if max_motif > 0:
        graph = nx.Graph()
        for i in range(0, num_nodes-1):
            for j in range(i+1, num_nodes):
                if adj_matrix[i, j] == 1:
                    graph.add_edge(i, j)
        nx.draw(graph)
        plt.show()
        motifs = load_motif_list([*range(1, max_motif + 1)])
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
                        motif_d_detailed, motif_d = compute_motifs(graph_mod,
                                                                   motifs=motifs,
                                                                   max_motif=max_motif)
                        distribution_list.append(motif_d)
                        distribution_list_detailed.append(motif_d_detailed)
                combination_list.append(x)
    else:
        (no_row, no_col) = np.where(con_matrix == 1)
        no_list = np.arange(no_row.shape[0])

        for r in tqdm(range(no_row.shape[0])):
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
                    motif_d_detailed, motif_d = compute_motifs(graph_mod,
                                                               motifs=motifs,
                                                               max_motif=max_motif)
                    distribution_list.append(motif_d)
                    distribution_list_detailed.append(motif_d_detailed)
                combination_list.append(x)

    if max_motif > 0:
        return combination_list, distribution_list_detailed, distribution_list
    else:
        return combination_list


def compute_motifs(graph, motifs, max_motif):
    hist = calc_motif_distribution(graph, motifs)
    hist = hist / sum(hist)
    # print(hist)

    hist_mod = []
    i_val = 0
    for _ in range(max_motif):
        hist_mod.append(sum(hist[i_val:MOTIF_COUNTS[_]]))
        i_val = _

    return hist, hist_mod


if __name__ == "__main__":
    # Graph creation
    num_initial_nodes = 25
    num_new_connections = 10
    adj_mat_init, G_init, con_mat = generate_graph(
        num_nodes=num_initial_nodes,
        num_connects=num_new_connections
    )
    nx.draw(G_init)
    plt.show()

    num_motifs = 4
    motif_list = load_motif_list([*range(1, num_motifs + 1)])  # load_motif_list([1, 2, 3, 4])
    motif_distribution_detailed, motif_distribution = compute_motifs(
        G_init, motifs=motif_list, max_motif=num_motifs
    )
    render_motif_hist(motif_distribution_detailed, motif_list)
    render_motif_hist(motif_distribution, motif_list)

    """"""
    adj_mat_mod, con_mat_mod = get_reduced_adjacency(
        adj_matrix=adj_mat_init, con_matrix=con_mat, num_nodes=1
    )
    # print(adj_mat_mod)

    config_list, distribution_list_detailed, distribution_list = get_possible_configurations(
        adj_matrix=adj_mat_mod, con_matrix=con_mat_mod, max_motif=num_motifs
    )
    print('Num configurations', len(config_list))

    dist = []
    for d in distribution_list:
        dist.append(w_d(d, motif_distribution))
    dist_det = []
    for d in distribution_list_detailed:
        dist_det.append(w_d(d, motif_distribution_detailed))

    plt.plot(dist)
    plt.plot(dist_det)
    plt.show()
    # for configuration in config_list:
    #     print(configuration)
    """"""
