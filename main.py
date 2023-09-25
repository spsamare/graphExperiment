import numpy as np
from itertools import combinations


def generate_graph(num_nodes=3):
    assert num_nodes > 1, "Number of nodes should be larger than one"

    up_triangle = np.random.randint(0, 2, num_nodes * (num_nodes - 1) // 2)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    adj_matrix[np.triu_indices(num_nodes, 1)] = up_triangle
    adj_matrix = adj_matrix + np.transpose(adj_matrix)

    node_deg = np.sum(adj_matrix, 0)
    # print(node_deg)
    node_index = np.argsort(node_deg)[::-1]
    # print(node_index)

    adj_matrix_new = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            adj_matrix_new[i, j] = adj_matrix[node_index[i], node_index[j]]

    adj_matrix_new = adj_matrix_new + np.transpose(adj_matrix_new)

    return adj_matrix_new


def get_reduced_adjacency(adj_matrix, num_nodes=1):
    assert adj_matrix.shape[0] - num_nodes > 1, "Reduced network should have at least two nodes"
    return adj_matrix[num_nodes:, num_nodes:]


def get_possible_configurations(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    up_triangle = adj_matrix[np.triu_indices(num_nodes, 1)]
    temp_mat = np.ones((num_nodes, num_nodes), dtype=int)
    temp_mat[np.triu_indices(num_nodes, 1)] = up_triangle
    (no_row, no_col) = np.where(temp_mat == 0)
    no_list = np.arange(no_row.shape[0])
    # print(adj_matrix)
    # print(no_row, no_col)

    combination_list = []
    for r in range(1, no_row.shape[0] + 1):
        for combination in combinations(no_list, r):
            x = adj_matrix.copy()
            for k in combination:
                x[no_row[k], no_col[k]] = 1
                x[no_col[k], no_row[k]] = 1
            combination_list.append(x)

    return combination_list


if __name__ == "__main__":
    adj_mat = generate_graph(num_nodes=10)
    # print(adj_mat)

    adj_mat = get_reduced_adjacency(adj_matrix=adj_mat, num_nodes=1)
    # print(adj_mat)

    config_list = get_possible_configurations(adj_matrix=adj_mat)
    print('Num configurations', len(config_list))
    # for configuration in config_list:
    #     print(configuration)
