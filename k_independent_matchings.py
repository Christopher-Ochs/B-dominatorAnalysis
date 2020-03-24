import networkx
import matplotlib.pyplot as plt
import networkx as nx
import random as rand
import statistics
import copy
import sys
import operator
from progress.bar import Bar
import utils


def show_graph_b_values(graph, pos, title):
    """
    Displays whe graph
    :param title: title to display
    :param graph: Networkx graph
    :param pos: position of nodes
    :return: None
    """
    plt.figure(1, figsize=(12, 12))
    labels = nx.get_node_attributes(graph, 'bValue')
    nx.draw_networkx(graph, pos, labels=labels, width=6, node_size=700)
    plt.title(str(title))
    plt.show()


def show_graph(graph, pos, title):
    """
    Displays whe graph
    :param title: title to display
    :param graph: Networkx graph
    :param pos: position of nodes
    :return: None
    """
    plt.figure(1, figsize=(12, 12))
    nx.draw_networkx(graph, pos, width=6, node_size=700)
    plt.title(str(title))
    plt.show()


def display_independent_results(average_recoveries: list, matching_cap):
    """
    Displayed the output of the independent
    :param average_recoveries:
    :param matching_cap:
    :return:
    """
    independent_matchings = [len(x) for x in average_recoveries]
    min_matchings = min(independent_matchings)
    print("\tAt least {} independent matchings found in all graphs when searching for {} matchings".format(min_matchings, matching_cap))
    print("\tThe average number of independent matchings per graph is {}".format(statistics.mean(independent_matchings)))

    for i in range(min_matchings):
        print("\tThe average recovery percentage for M{} = {}".format(i + 1, statistics.mean(x[i] for x in average_recoveries)))
    print("\n\n")


def test_graph(b_values_graph):
    # Perform the matching
    matching_graph = utils.perform_b_matching(b_values_graph)

    # Save the actual weights achieved from the algorithm
    actual_weights = utils.get_total_weights(matching_graph)

    return matching_graph, actual_weights


def test_k_graph_independence(input_graph, min_weight, max_weight, matching_cap=2, results=False, display=False):
    utils.add_weights_to_graph(input_graph, min_weight, max_weight)
    goal_weights = utils.get_total_weights(input_graph)
    b_value_graph = utils.add_b_values_to_graph(input_graph)

    pos = nx.spring_layout(input_graph)

    average_recoveries = []
    matching_count = 0

    while matching_count < matching_cap and (not nx.is_empty(b_value_graph)) and nx.is_connected(b_value_graph):
        starting_graph = copy.deepcopy(b_value_graph)

        matching_count += 1
        matching_graph, actual_weights = test_graph(b_value_graph)
        average_recoveries.append(actual_weights / goal_weights)

        edges_to_remove = []
        for u, v in matching_graph.edges:
            if matching_graph.get_edge_data(u, v)['weight'] != 0:
                edges_to_remove.append((u, v))

        starting_graph.remove_edges_from(edges_to_remove)
        b_value_graph = starting_graph

        if display:
            show_graph(b_value_graph, pos, "Node Labels   Iteration " + str(matching_count))
            show_graph_b_values(b_value_graph, pos, "b-Values   Iteration " + str(matching_count))

    if results:
        print("========== RESULTS ==========")
        print("Final matching achieved = " + str(matching_count))
        print("Average Recovery Percentage = " + str(average_recovery))
        print("Best Recovery Percentage = " + str(max(average_recoveries)))
        print("Worst Recovery Percentage = " + str(min(average_recoveries)))

    return matching_count, average_recoveries


def test_k_graph_independence_dominating(input_graph, min_weight, max_weight, matching_cap=2, results=False, display=False):
    utils.add_weights_to_graph(input_graph, min_weight, max_weight)
    goal_weights = utils.get_total_weights(input_graph)
    b_value_graph = utils.add_b_values_to_graph(input_graph)

    # Find vertex with maximum weight and double its bValue
    node, _ = utils.get_maximum_b_value(b_value_graph)
    value = int(utils.sum_neighbors_b_values(b_value_graph, node) * 1.1)
    b_value_graph.nodes[node]['bValue'] += value
    goal_weights += (value // 2)

    pos = nx.spring_layout(input_graph)

    average_recoveries = []
    matching_count = 0

    while matching_count < matching_cap and (not nx.is_empty(b_value_graph)) and nx.is_connected(b_value_graph):
        starting_graph = copy.deepcopy(b_value_graph)

        matching_count += 1
        matching_graph, actual_weights = test_graph(b_value_graph)
        average_recoveries.append(actual_weights / goal_weights)

        edges_to_remove = []
        for u, v in matching_graph.edges:

            if matching_graph.get_edge_data(u, v)['weight'] != 0:
                edges_to_remove.append((u, v))

        starting_graph.remove_edges_from(edges_to_remove)
        b_value_graph = starting_graph

        if display:
            show_graph(b_value_graph, pos, "Node Labels   Iteration " + str(matching_count))
            show_graph_b_values(b_value_graph, pos, "b-Values   Iteration " + str(matching_count))

    if results:
        print("========== RESULTS ==========")
        print("Final matching achieved = " + str(matching_count))
        print("Average Recovery Percentage = " + str(average_recovery))
        print("Best Recovery Percentage = " + str(max(average_recoveries)))
        print("Worst Recovery Percentage = " + str(min(average_recoveries)))

    return matching_count, average_recoveries


def test_complete_graph_independence(trials=1000, matching_cap=4, nodes_set=None, min_weight=25, max_weight=2500):
    print("ANALYZING COMPLETE GRAPH:\n")

    if nodes_set is None:
        nodes_set = [50, 75, 100, 150, 200, 250]

    for nodes in nodes_set:
        average_recoveries = []
        progress_bar = Bar("\tTesting Complete Graph N={}".format(nodes), max=trials, stream=sys.stdout)

        for trial in range(trials):
            progress_bar.next()
            graph = utils.generate_complete_graph(nodes)
            independence_count, average_recovery = test_k_graph_independence(graph, min_weight, max_weight, matching_cap=matching_cap)
            average_recoveries.append(average_recovery)

        progress_bar.finish()
        display_independent_results(average_recoveries, matching_cap)


def test_random_prob_graph_independence(trials=1000, matching_cap=2, nodes=100, prob_set=None, min_weight=25, max_weight=2500):
    print("ANALYZING RANDOM PROBABILITY GRAPH:\n")

    if prob_set is None:
        prob_set = [0.4, 0.5, 0.6, 0.7, 0.8]

    for prob in prob_set:
        average_recoveries = []
        progress_bar = Bar("\tTesting Random Probability Graph N={} P={}.format(nodes, prob)", max=trials, stream=sys.stdout)

        for trial in range(trials):
            progress_bar.next()
            graph = utils.generate_random_prob_graph(nodes, prob)
            independence_count, average_recovery = test_k_graph_independence(graph, min_weight, max_weight, matching_cap=matching_cap)
            average_recoveries.append(average_recovery)

        progress_bar.finish()
        display_independent_results(average_recoveries, matching_cap)


def test_random_edge_graph_independence(trials=1000, matching_cap=4, nodes=100, average_degrees=None, min_weight=25, max_weight=2500):
    print("ANALYZING RANDOM EDGE GRAPH:\n")

    if average_degrees is None:
        average_degrees = [3, 4, 5, 10, 20, 25, 45]

    edge_list = [x * nodes for x in average_degrees]

    for edges in edge_list:
        average_recoveries = []
        progress_bar = Bar("\tTesting Random Edge Graph N={} E={}".format(nodes, edges), max=trials, stream=sys.stdout)

        for trial in range(trials):
            progress_bar.next()
            graph = utils.generate_random_edge_graph(nodes, edges)
            independence_count, average_recovery = test_k_graph_independence(graph, min_weight, max_weight, matching_cap=matching_cap)
            average_recoveries.append(average_recovery)

        progress_bar.finish()
        display_independent_results(average_recoveries, matching_cap)


def test_barabasi_albert_graph_independence(trials=1000, matching_cap=4, nodes=100, new_degree=None, min_weight=25, max_weight=2500):
    print("ANALYZING BARABASI GRAPH:\n")

    if new_degree is None:
        new_degree = [2, 4, 6, 8, 10, 15, 20]

    for degree in new_degree:
        average_recoveries = []
        progress_bar = Bar("\tTesting Barabasi Albert Graph N={} Attachment={}".format(nodes, degree), max=trials, stream=sys.stdout)

        for trial in range(trials):
            progress_bar.next()
            graph = utils.generate_barabasi_pref_graph(nodes, degree)
            independence_count, average_recovery = test_k_graph_independence(graph, min_weight, max_weight, matching_cap=matching_cap)
            average_recoveries.append(average_recovery)

        progress_bar.finish()
        display_independent_results(average_recoveries, matching_cap)


def test_small_world_graph_independence(trials=1000, matching_cap=4, nodes=100, initial_degrees=None, rewiring_probabilities=None, min_weight=25, max_weight=2500):
    print("ANALYZING SMALL WORLD GRAPH:\n")

    if initial_degrees is None:
        initial_degrees = [3, 5, 10, 15, 20, 25]

    if rewiring_probabilities is None:
        rewiring_probabilities = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    for initial_degree in initial_degrees:

        for rewiring_probability in rewiring_probabilities:
            average_recoveries = []
            progress_bar = Bar("\tTesting Small World Graph Initial Degree={} Rewiring Probability={}".format(initial_degree, rewiring_probability), max=trials, stream=sys.stdout)

            for trial in range(trials):
                progress_bar.next()
                graph = utils.generate_small_world(nodes, initial_degree, rewiring_probability)
                _, average_recovery = test_k_graph_independence(graph, min_weight, max_weight, matching_cap=matching_cap)
                average_recoveries.append(average_recovery)

            progress_bar.finish()
            display_independent_results(average_recoveries, matching_cap)


def test_near_complete_independence(trials=1000, matching_cap=4, node_set=None, min_weight=25, max_weight=2500):
    print("ANALYZING NEAR COMPLETE GRAPH:\n")

    if node_set is None:
        node_set = [50, 75, 100, 150, 200, 250]

    for nodes in node_set:
        num_edges = ((nodes * (nodes - 1)) // 2) - 1
        average_recoveries = []
        progress_bar = Bar("\tTesting near-complete Graph with {} nodes and {} edges".format(nodes, num_edges), max=trials, stream=sys.stdout)

        for i in range(trials):
            progress_bar.next()
            graph = utils.generate_random_edge_graph(nodes, num_edges)
            _, average_recovery = test_k_graph_independence(graph, min_weight, max_weight, matching_cap=matching_cap)
            average_recoveries.append(average_recovery)

        progress_bar.finish()
        display_independent_results(average_recoveries, matching_cap)


def test_dominating_independence(trials=1000, matching_cap=4, num_nodes=None, min_weight=25, max_weight=2500):
    print("ANALYZING GRAPH WITH DOMINATING VERTEX:\n")

    if num_nodes is None:
        num_nodes = [50, 75, 100, 150, 200, 250]

    for nodes in num_nodes:
        average_recoveries = []
        progress_bar = Bar("\tTesting complete graph with {} nodes".format(nodes), max=trials, stream=sys.stdout)

        for i in range(trials):
            progress_bar.next()
            graph = utils.generate_complete_graph(nodes)
            _, average_recovery = test_k_graph_independence_dominating(graph, min_weight, max_weight, matching_cap=matching_cap)
            average_recoveries.append(average_recovery)

        progress_bar.finish()
        display_independent_results(average_recoveries, matching_cap)
        average_recoveries.clear()

        for num_edges in [x * nodes for x in [3, 5, 10, 25, 45]]:
            average_recoveries = []
            progress_bar = Bar("\tTesting graph with {} vertices and {} edges".format(nodes, num_edges), max=trials, stream=sys.stdout)

            for i in range(trials):
                progress_bar.next()
                edge_graph = utils.generate_random_edge_graph(nodes, num_edges)
                _, average_recovery = test_k_graph_independence_dominating(edge_graph, min_weight, max_weight, matching_cap=matching_cap)
                average_recoveries.append(average_recovery)

            progress_bar.finish()
            display_independent_results(average_recoveries, matching_cap)



def test_ring_graph_independence(num_nodes=None, initial_degrees=None, min_weight=25, max_weight=2500, trials=1000, matching_cap=4):
    print("ANALYZING RING GRAPH:\n")

    if initial_degrees is None:
        initial_degrees = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]

    if num_nodes is None:
        num_nodes = [100]

    for nodes in num_nodes:

        for initial_degree in initial_degrees:
            average_recoveries = []
            progress_bar = Bar("\tTesting ring Graph Initial Degree={} and vertices={}".format(initial_degree, nodes), max=trials, stream=sys.stdout)

            for trial in range(trials):
                progress_bar.next()
                ring_graph = utils.generate_small_world(nodes, initial_degree, 0)
                _, average_recovery = test_k_graph_independence(ring_graph, min_weight, max_weight, matching_cap=matching_cap)
                average_recoveries.append(average_recovery)

            progress_bar.finish()
            display_independent_results(average_recoveries, matching_cap)


def main():
    t = 3
    test_complete_graph_independence(trials=t)
    test_random_edge_graph_independence(trials=t)
    test_random_prob_graph_independence(trials=t)
    test_barabasi_albert_graph_independence(trials=t)
    test_small_world_graph_independence(trials=t)
    test_near_complete_independence(trials=t)
    test_dominating_independence(trials=t)
    test_ring_graph_independence(trials=t)


if __name__ == "__main__":
    main()
