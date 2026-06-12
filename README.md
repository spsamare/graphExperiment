# Graph Experiments

The focus is to generate a random graph with undirected links with the nodes that are indexed from the highest to the lowest degree. Then all possible configurations that could be obtained by only adding new connections are to be found.

| Function                                       | Use                                                                       | Output                                                                      |
|------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `generate_graph(num_nodes)`                    | Generate a graph with `num_nodes` number of nodes                         | Adjacency matrix `adj_matrix` indexed from the highest to the lowers degree |
| `get_possible_configurations(adj_matrix)`      | Find all possible configurations in terms of adding new links             | A list of new adjacency matrices `combination_list`                         |
| `get_reduced_adjacency(adj_matrix, num_nodes)` | Remove `num_nodes` nodes with the highest degrees from the existing graph | Adjacency matrix `adj_matrix` of the new graph                              |


Graph motif generation is from [here](https://github.com/TianYafu/RSG_Graph_Motif_Counter).
