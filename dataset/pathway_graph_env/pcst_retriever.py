import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from pcst_fast import pcst_fast

from dataset.pathway_graph_env.pathway_graph_api_utils import node_to_name, edge_to_name


# The old pcst_retriever version
def binary_line_search(f, y, low, high, epsilon=1e-6):
    """
    Searches for the closest variable X to make f(x) close to the target y,
    assuming f(x) is monotonically increasing.

    :param f: A monotonically increasing function
    :param y: The target value
    :param low: The lower bound for the search
    :param high: The upper bound for the search
    :param epsilon: The tolerance for the search (default is 1e-6)
    :return: The closest variable X to make f(x) close to the target y
    """
    while high - low > epsilon:
        mid = (low + high) / 2
        if f(mid) == y:
            break
        elif f(mid) > y:
            low = mid
        else:
            high = mid

    return (low + high) / 2, low, high


def pcst_retrieval(edges_array, num_nodes, n_prizes, e_prizes, cost_e=0.5, num_clusters=1):
    root = -1  # unrooted
    pruning = 'gw'
    verbosity_level = 0

    costs = []
    edges_true = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(edges_array):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges_true)] = i
            edges_true.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges_true)
    # if len(virtual_costs) > 0:
    costs = np.array(costs + virtual_costs)
    edges = np.array(edges_true + virtual_edges)

    vertices, edges_id = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < num_nodes]
    selected_edges = [mapping_e[e] for e in edges_id if e < num_edges]
    virtual_vertices = vertices[vertices >= num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)

    edge_index = edges_array[list(np.unique(selected_edges)), :]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[:, 0], edge_index[:, 1]]))
    return selected_nodes, edge_index


def pcst_retrieval_size(edges_array, num_nodes, n_prizes, e_prizes, target_size, num_clusters=1):
    f = lambda c: len(
        pcst_retrieval(edges_array, num_nodes, n_prizes, e_prizes, cost_e=c, num_clusters=num_clusters)[1])
    cost_mid, cost_low, cost_high = binary_line_search(f, target_size, 0.005, 5, 1e-8)
    return pcst_retrieval(edges_array, num_nodes, n_prizes, e_prizes, cost_e=cost_mid,
                          num_clusters=num_clusters), cost_mid


def plot_pcst_result(origin_graph, pcst_subgraph, all_hsa_graph, all_entry, node_prizes, edge_prizes, key_words_string):
    G_merged = nx.compose(pcst_subgraph, origin_graph)
    plt.clf()
    # Draw the parts we want
    # Choose colors: 'red' for the special node, 'blue' for the others
    node_colors = []
    node_size = []
    for node in G_merged.nodes:
        if node in pcst_subgraph.nodes and node in origin_graph.nodes:
            color = 'green'
        elif node in pcst_subgraph.nodes:
            color = 'blue'
        elif node in origin_graph.nodes:
            color = 'red'
        else:
            raise ValueError('node not in graph')
        rgba_color = to_rgba(color, alpha=node_prizes[node])
        node_colors.append(rgba_color)
        node_size.append(60 * node_prizes[node])
    edge_colors = []
    edge_size = []
    for edge in G_merged.edges:
        if edge in pcst_subgraph.edges and edge in origin_graph.edges:
            color = 'green'
        elif edge in pcst_subgraph.edges:
            color = 'blue'
        elif edge in origin_graph.edges:
            color = 'red'
        else:
            raise ValueError('node not in graph')
        rgba_color = to_rgba(color, alpha=edge_prizes[edge])
        edge_colors.append(rgba_color)
        edge_size.append(3 * edge_prizes[edge])
    node_labels = {}
    for node in G_merged.nodes:
        node_labels[node] = node_to_name(node, all_hsa_graph, all_entry)
    edge_labels = {}
    for edge in G_merged.edges:
        edge_labels[edge] = edge_to_name(edge, all_hsa_graph, all_entry)
    layout = nx.kamada_kawai_layout(G_merged, scale=10)
    # Adjust edge label positions
    edge_label_positions = {}
    for edge, label_pos in layout.items():
        edge_label_positions[edge] = label_pos + 1

    nx.draw_networkx_nodes(G_merged, pos=layout, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_edges(G_merged, pos=layout, edge_color=edge_colors, width=edge_size)
    nx.draw_networkx_labels(G_merged, pos=layout, labels=node_labels, font_size=1, font_weight='bold')
    nx.draw_networkx_edge_labels(G_merged, pos=layout, edge_labels=edge_labels, font_size=1,
                                 verticalalignment='top', rotate=False
                                 )
    if not os.path.exists(os.path.join('plot_result', 'pcst')):
        os.makedirs(os.path.join('plot_result', 'pcst'))
    plt.savefig(os.path.join('plot_result', 'pcst', '{}.pdf'.format(key_words_string)), format='pdf',
                bbox_inches='tight', dpi=300)
    plt.clf()
