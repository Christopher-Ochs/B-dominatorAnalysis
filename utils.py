import matplotlib.pyplot as plt
import networkx as nx
import random as rand
import copy
import operator


def generate_random_prob_graph(num_nodes, prob):
    """
    Generates a random probability graph and removes isolated nodes
    :param num_nodes: number of nodes in the graph
    :param prob: probability every pair of nodes will create an edge
    :return: Graph
    """
    graph = nx.gnp_random_graph(num_nodes, prob)
    graph.remove_nodes_from(list(nx.isolates(graph)))

    while not nx.is_connected(graph):
        graph = nx.gnp_random_graph(num_nodes, prob)
        graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph


def generate_random_edge_graph(num_nodes, num_edges):
    """
    Generates a networkx graph and removes any isolated nodes in the graph
    :param num_nodes: number of nodes in the generated graph
    :param num_edges: number of edges in the generated graph
    :return: Graph
    """
    graph = nx.dense_gnm_random_graph(num_nodes, num_edges)
    graph.remove_nodes_from(list(nx.isolates(graph)))

    while not nx.is_connected(graph):
        graph = nx.dense_gnm_random_graph(num_nodes, num_edges)
        graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph


def generate_barabasi_pref_graph(num_nodes, attachments):
    """
    Generates a networkx graph according to Barabasi Albert's preferential attachment
    :param num_nodes: number of nodes in the generated graph
    :param attachments: number of edges to create when a new node is added
    :return: Graph
    """
    return nx.barabasi_albert_graph(num_nodes, attachments)


def generate_complete_graph(num_nodes):
    """
    Generates a complete networkx graph
    :param num_nodes: number of nodes in the generated complete graph
    :return: Graph
    """
    return nx.complete_graph(num_nodes)


def generate_small_world(num_nodes, starting_degree, rewire_probability):
    """
    Generates a small world graph.
    :param num_nodes: number of nodes
    :param starting_degree: initial degree of each node
    :param rewire_probability: probability that each edge is destroyed and a new neighbor is selected. The criteria for
        new neighbor selection can be seen here:
    :return: Graph
    """
    return nx.connected_watts_strogatz_graph(num_nodes, starting_degree, rewire_probability)


def add_weights_to_graph(graph, weight_min, weight_max):
    """
    Adds random weights weights to the edges on the graph and returns the graph.
    :param graph: Graph to add weights to
    :param weight_min: Minimum weight value allowed for edge
    :param weight_max: Maximum weight value allowed for edge
    :return: The same graph with weights added to edges
    """
    rand.seed()

    for (u, v, w) in graph.edges.data():
        graph.add_edge(u, v, weight=rand.randint(weight_min, weight_max))

    return copy.deepcopy(graph)


def add_b_values_to_graph(graph):
    """
    :param graph: Networkx graph to add b-values to, must have weighted edges
    :return: a new graph with b-values added
    """
    # We don't want to modify the original graph so make a copy first
    g = copy.deepcopy(graph)
    nx.set_node_attributes(g, 0, 'bValue')
    weights = nx.get_edge_attributes(g, 'weight')

    for (node1, node2), w in weights.items():
        g.add_node(node1, bValue=(w + get_b_value(g, node1)))
        g.add_node(node2, bValue=(w + get_b_value(g, node2)))

    nx.set_edge_attributes(g, 0, 'weight')

    return g


def show_graph_with_b_values(graph, pos):
    """
    Displays the graph with b values on each node
    :param graph: Networkx Graph with param 'bValue' as node attribute
    :param pos: Position of each node
    :return: None
    """
    plt.figure(1, figsize=(12, 12))
    labels = nx.get_node_attributes(graph, 'bValue')
    nx.draw_networkx(graph, pos, labels=labels, width=6, node_size=700)
    plt.show()


def show_graph(graph, pos):
    """
    Displays whe graph; may contain weighted edges
    :param graph: Networkx graph
    :param pos: position of nodes
    :return: None
    """
    plt.figure(1, figsize=(12, 12))
    nx.draw_networkx_nodes(graph, pos, node_size=700)
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family='sans-serif')
    nx.draw_networkx_edges(graph, pos, width=6)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


def get_b_value(graph, node):
    """
    :param graph: Networkx graph
    :param node: Node
    :return: the value of attribute 'bValue' on the node
    """
    return graph.nodes[node]['bValue']


def get_maximum_b_value(graph):
    """
    :param graph: Networkx graph with node attribute bValues
    :return: node id and value of maximum bValue
    """
    node_attribute_dict = nx.get_node_attributes(graph, "bValue")

    return max(node_attribute_dict.items(), key=operator.itemgetter(1))


def sum_neighbors_b_values(graph, node):
    """
    :param graph: Networkx graph with bValues
    :param node: node id
    :return: sum of node's neighbors bValues
    """
    neighbors_sum = 0

    for n in graph.neighbors(node):
        neighbors_sum += get_b_value(graph, n)

    return neighbors_sum


def get_eta_value(graph, node):
    """
    Returns the subtraction of the b value of the node from the summation of the b values of neighboring nodes
    :param graph: Networkx graph
    :param node: Node
    :return: the eta value
    """

    return sum_neighbors_b_values(graph, node) - get_b_value(graph, node)


def get_mu_value(graph, node):
    """
    :param graph: Networkx graph
    :param node: Node
    :return: the minimum value of the eta value of the neighbors of node
    """
    candidate_values = []

    for n in graph.neighbors(node):
        candidate_values.append(get_eta_value(graph, n))

    return min(candidate_values)


def has_neighbor_with_b_value(graph, node):
    """
    :param graph: Networkx graph with bValues
    :param node: Node id
    :return: true if any neighbor has bValue; else false
    """
    for n in graph.neighbors(node):

        if get_b_value(graph, n) != 0:
            return True

    return False


def find_eta_lt0(graph):
    """
    finds a node with a eta value less than 0; if none exists returns negative 1
    NOTE: We don't want to cover the case where all neighbors bValues are 0
    :param graph: Networkx Graph
    :return: node or -1
    """
    for n in graph.nodes():

        if get_b_value(graph, n) != 0 and get_eta_value(graph, n) <= 0 and has_neighbor_with_b_value(graph, n):
            return n

    return -1


def find_eta_e0(graph):
    """
    finds a node with eta value equal to 0; if none exists returns negative 1
    :param graph: Networkx Graph
    :return: node or -1
    """
    for n in graph.nodes():
        # We must ensure that we no longer count nodes whose b value is 0
        if get_b_value(graph, n) != 0 and get_eta_value(graph, n) == 0:
            return n

    return -1


def find_candidate_edge(graph):
    """
    Enumerates through the edges of the graph and finds two nodes whose b-value are not 0.
    :param graph: Networkx graph
    :return: the pair of nodes or -1
    """
    for u, v in graph.edges():

        if get_b_value(graph, u) != 0 and get_b_value(graph, v) != 0:
            return u, v

    return -1, -1


def block_exchange(original_graph, matching_graph, node1, node2, value):
    """
    Reduces the b-value of each node by the param value and adds to matching
    :param matching_graph:
    :param original_graph:
    :param node1: A node, must share edge with node 2
    :param node2: A node, must share edge with node 1
    :param value: The value to increase the edge weight and reduce the b-values
    :return: None
    """
    if original_graph.nodes[node1]['bValue'] < value or original_graph.nodes[node2]['bValue'] < value:
        raise Exception(
            'Value should not exceed b-value of either node. Value was {}, Node1 b-value {}, Node2 b-value {}'.format(
                value, original_graph.nodes[node1]['bValue'], original_graph.nodes[node2]['bValue']))

    original_graph.nodes[node1]['bValue'] -= value
    original_graph.nodes[node2]['bValue'] -= value

    if original_graph.nodes[node1]['bValue'] == 0:
        original_graph.remove_node(node1)

    if original_graph.nodes[node2]['bValue'] == 0:
        original_graph.remove_node(node2)

    if matching_graph.has_edge(node1, node2):
        matching_graph.get_edge_data(node1, node2)['weight'] += value
    else:
        matching_graph.add_edge(node1, node2, weight=value)


def get_total_weights(graph):
    """
    Returns the total weights of the edges of the graph.
    :param graph: Networkx graph
    :return: total weight of graph
    """
    total = 0

    for u, v in graph.edges:
        total += graph.get_edge_data(u, v)['weight']

    return total


def perform_b_matching(original_graph):
    """
    Performs a b-matching on the provided graph with b-values on the nodes.
    :param original_graph: Networkx graph
    :return: a graph with weights on edges assigned to from b-values
    """
    # Ensure that no weights exists on edges before beginning
    matching_graph = nx.Graph()

    # Find any dominating nodes and perform the block exchange with all its neighbors
    node = find_eta_lt0(original_graph)

    while node != -1:
        neighbors = copy.deepcopy(original_graph.neighbors(node))

        for n in neighbors:
            block_exchange(original_graph, matching_graph, n, node, get_b_value(original_graph, n))

        # Search for new dominating node
        node = find_eta_lt0(original_graph)

    # Find a candidate edge such that the b-value of both nodes is non zero and continue while this is true
    (u, v) = find_candidate_edge(original_graph)

    while u != -1:

        # Get the b-value and mu value for each node in the edge
        b_u = get_b_value(original_graph, u)
        b_v = get_b_value(original_graph, v)
        mu_u = get_mu_value(original_graph, u) // 2
        mu_v = get_mu_value(original_graph, v) // 2

        # Select the smallest value greater than 0 and perform the block exchange with that value
        value = min(i for i in [b_u, b_v, mu_u, mu_v] if i > 0)
        block_exchange(original_graph, matching_graph, u, v, value)

        # If there is a node whose neighbors b-values equal it's b-value perform the exchange with all neighbors.
        node = find_eta_e0(original_graph)

        while node != -1:
            neighbors = copy.deepcopy(original_graph.neighbors(node))

            for n in neighbors:
                block_exchange(original_graph, matching_graph, node, n, get_b_value(original_graph, n))

            node = find_eta_e0(original_graph)

        # Search for new candidate edge
        (u, v) = find_candidate_edge(original_graph)

    return matching_graph
