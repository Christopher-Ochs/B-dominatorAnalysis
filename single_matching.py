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


def display_single_matching_results(approx_actual: list, approx_goal: list):
    """
    Outputs the results of a single matching
    :param approx_actual: the sum of the b-values of the b-matching found by the algorithm
    :param approx_goal: the sum of the b-values of the perfect
    :return:
    """
    individual = [i / j for i, j in zip(approx_actual, approx_goal)]
    worst_case = min(individual)
    best_case = max(individual)
    average_goal = statistics.mean(approx_goal)
    average_actual = statistics.mean(approx_actual)
    recovery_percentage = average_actual / average_goal
    print("The average recovery percentage: {:>2.10}".format(recovery_percentage, 6))
    print("The best recovery percentage: {:>2.15}".format(round(best_case, 6)))
    print("The worst recovery percentage: {:>2.14}".format(round(worst_case, 6)))
    print("Perfect Matchings found: {}\n\n".format(individual.count(1)))


def test_algorithm(input_graph, min_weight, max_weight):
    """
    :param input_graph: a skeleton graph with vertices and edges
    :param min_weight: minimum weight to add to an edge
    :param max_weight: maximum weight to add to an edge
    :return: the actual weight of the matching found and the goal matching.
    """

    # Add weights to the random graph.
    weighted_graph = utils.add_weights_to_graph(input_graph, min_weight, max_weight)

    # Save the goal weights; This is the perfect matching goal.
    goal_weights = utils.get_total_weights(weighted_graph)

    # Add b-values to the graph and remove wights
    b_graph = utils.add_b_values_to_graph(weighted_graph)

    # Perform the matching; adds weights to edges calculated from b-values
    matching_graph = utils.perform_b_matching(b_graph)

    # Save the actual weights achieved from the algorithm
    actual_weights = utils.get_total_weights(matching_graph)

    return actual_weights, goal_weights


def test_algorithm_with_dominating(input_graph, min_weight, max_weight):
    # Add weights to the random graph.
    weighted_graph = utils.add_weights_to_graph(input_graph, min_weight, max_weight)

    # Save the goal weights; This is the perfect matching goal.
    goal_weights = utils.get_total_weights(weighted_graph)

    # Add b-values to the graph and remove wights
    b_graph = utils.add_b_values_to_graph(weighted_graph)

    # Find vertex with maximum weight and double its bValue
    node, _ = utils.get_maximum_b_value(b_graph)
    value = int(utils.sum_neighbors_b_values(b_graph, node) * 1.1)
    b_graph.nodes[node]['bValue'] += value
    goal_weights += (value // 2)

    # Perform the matching; adds weights to edges calculated from b-values
    matching_graph = utils.perform_b_matching(b_graph)

    # Save the actual weights achieved from the algorithm
    actual_weights = utils.get_total_weights(matching_graph)

    return actual_weights, goal_weights


def analyze_complete_graph(num_nodes=None, min_weight=25, max_weight=2500, trials=1000):
    print("ANALYZING COMPLETE GRAPH:\n")

    if num_nodes is None:
        num_nodes = [50, 100, 150, 200]

    # For each average degree run the b-matching algorithm
    for nodes in num_nodes:
        approx_actual = []
        approx_goal = []

        progress_bar = Bar("Complete graph with {} vertices".format(nodes), max=trials, stream=sys.stdout)
        for i in range(trials):
            complete_graph = utils.generate_complete_graph(nodes)
            actual_weights, goal_weights = test_algorithm(complete_graph, min_weight, max_weight)
            progress_bar.next()

            approx_actual.append(actual_weights)
            approx_goal.append(goal_weights)

        progress_bar.finish()
        display_single_matching_results(approx_actual, approx_goal)


def analyze_random_prob_graph(num_nodes=100, prob_set=None, min_weight=25, max_weight=2500, trials=1000):
    print("ANALYZING RANDOM PROBABILITY GRAPH:\n")

    if prob_set is None:
        prob_set = [0.4, 0.5, 0.6, 0.7, 0.8]

    # For each probability
    for prob in prob_set:
        approx_actual = []
        approx_goal = []
        progress_bar = Bar("Graph with {} vertices and link probability of {}".format(num_nodes, prob), max=trials, stream=sys.stdout)

        for i in range(trials):
            prob_graph = utils.generate_random_prob_graph(num_nodes, prob)
            actual_weights, goal_weights = test_algorithm(prob_graph, min_weight, max_weight)
            progress_bar.next()
            approx_actual.append(actual_weights)
            approx_goal.append(goal_weights)

        progress_bar.finish()
        display_single_matching_results(approx_actual, approx_goal)


def analyze_random_edge_graph(num_nodes=100, edge_set=None, min_weight=25, max_weight=2500, trials=1000):
    print("ANALYZING RANDOM EDGE GRAPH:\n")

    if edge_set is None:
        # Generate number of edges; with average degrees of 3, 4, 5, 10, 20, 25, 45
        edge_set = [x * num_nodes for x in [3, 4, 5, 10, 20, 25, 45]]

    # For each average degree run the b-matching algorithm
    for num_edges in edge_set:
        approx_actual = []
        approx_goal = []
        progress_bar = Bar("Graph with {} vertices and {} edges".format(str(num_nodes), str(num_edges)), max=trials, stream=sys.stdout)

        for i in range(trials):
            edge_graph = utils.generate_random_edge_graph(num_nodes, num_edges)
            actual_weights, goal_weights = test_algorithm(edge_graph, min_weight, max_weight)
            progress_bar.next()
            approx_actual.append(actual_weights)
            approx_goal.append(goal_weights)

        progress_bar.finish()
        display_single_matching_results(approx_actual, approx_goal)


def analyze_barabasi_graph(num_nodes=100, new_degree=None, min_weight=25, max_weight=2500, trials=1000):
    print("ANALYZING BARABASI GRAPH:\n")

    if new_degree is None:
        new_degree = [2, 4, 6, 8, 10, 15, 20]

    for attachment in new_degree:
        approx_actual = []
        approx_goal = []
        progress_bar = Bar("Barabasi graph with {} vertices and {} attachments".format(num_nodes, attachment), max=trials, stream=sys.stdout)

        for i in range(trials):
            pref_graph = utils.generate_barabasi_pref_graph(num_nodes, attachment)
            actual_weights, goal_weights = test_algorithm(pref_graph, min_weight, max_weight)
            progress_bar.next()
            approx_actual.append(actual_weights)
            approx_goal.append(goal_weights)

        progress_bar.finish()
        display_single_matching_results(approx_actual, approx_goal)


def analyze_small_world_graph(num_nodes=100, initial_degrees=None, rewiring_probabilities=None, min_weight=25, max_weight=2500, trials=1000):
    print("ANALYZING SMALL WORLD GRAPH:\n")

    if initial_degrees is None:
        initial_degrees = [3, 5, 10, 15, 20, 25]

    if rewiring_probabilities is None:
        rewiring_probabilities = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    for initial_degree in initial_degrees:
        for rewiring_probability in rewiring_probabilities:
            approx_actual = []
            approx_goal = []
            progress_bar = Bar("Small World with initial degree of {} and rewiring probability of {}".format(initial_degree, rewiring_probability), max=trials, stream=sys.stdout)

            for i in range(trials):
                small_world = utils.generate_small_world(num_nodes, initial_degree, rewiring_probability)
                actual_weights, goal_weights = test_algorithm(small_world, min_weight, max_weight)
                progress_bar.next()
                approx_actual.append(actual_weights)
                approx_goal.append(goal_weights)

            progress_bar.finish()
            display_single_matching_results(approx_actual, approx_goal)


def analyze_near_complete(node_set=None, min_weight=25, max_weight=2500, trials=1000):
    print("ANALYZING NEAR COMPLETE GRAPH:\n")

    if node_set is None:
        node_set = [50, 75, 100, 150, 200, 250]

    for nodes in node_set:
        edges = ((nodes * (nodes - 1)) // 2) - 1
        approx_actual = []
        approx_goal = []
        progress_bar = Bar("Near Complete Graph with {} nodes and {} edges".format(nodes, edges), max=trials, stream=sys.stdout)

        for i in range(trials):
            near_complete_graph = utils.generate_random_edge_graph(nodes, edges)
            actual_weights, goal_weights = test_algorithm(near_complete_graph, min_weight, max_weight)
            progress_bar.next()
            approx_actual.append(actual_weights)
            approx_goal.append(goal_weights)

        progress_bar.finish()
        display_single_matching_results(approx_actual, approx_goal)


def analyze_dominating(num_nodes=None, min_weight=25, max_weight=2500, trials=1000):
    print("ANALYZING GRAPH WITH DOMINATING VERTEX:\n")

    if num_nodes is None:
        num_nodes = [50, 75, 100, 150, 200, 250]

    for nodes in num_nodes:
        approx_actual = []
        approx_goal = []

        progress_bar = Bar("Complete graph with {} nodes".format(nodes), max=trials, stream=sys.stdout)
        for i in range(trials):
            complete_graph = utils.generate_complete_graph(nodes)
            actual_weights, goal_weights = test_algorithm_with_dominating(complete_graph, min_weight, max_weight)
            progress_bar.next()

            approx_actual.append(actual_weights)
            approx_goal.append(goal_weights)

        progress_bar.finish()
        display_single_matching_results(approx_actual, approx_goal)

        for num_edges in [x * nodes for x in [3, 5, 10, 25, 45]]:
            approx_actual = []
            approx_goal = []
            progress_bar = Bar("Graph with {} vertices and {} edges".format(nodes, num_nodes), max=trials, stream=sys.stdout)

            for i in range(trials):
                edge_graph = utils.generate_random_edge_graph(nodes, num_edges)
                actual_weights, goal_weights = test_algorithm_with_dominating(edge_graph, min_weight, max_weight)
                progress_bar.next()
                approx_actual.append(actual_weights)
                approx_goal.append(goal_weights)

            progress_bar.finish()
            display_single_matching_results(approx_actual, approx_goal)


def analyze_ring_graph(num_nodes=None, initial_degrees=None, min_weight=25, max_weight=2500, trials=1000):
    print("ANALYZING RING GRAPH:\n")

    if initial_degrees is None:
        initial_degrees = [2, 3, 4, 5, 6, 7, 8]

    if num_nodes is None:
        num_nodes = [100]

    for nodes in num_nodes:
        for initial_degree in initial_degrees:

            approx_actual = []
            approx_goal = []
            progress_bar = Bar("Analyzing ring graph with initial degree = {} and vertices = {}".format(initial_degree, nodes), max=trials, stream=sys.stdout)

            for i in range(trials):
                small_world = utils.generate_small_world(nodes, initial_degree, 0)
                actual_weights, goal_weights = test_algorithm(small_world, min_weight, max_weight)
                progress_bar.next()
                approx_actual.append(actual_weights)
                approx_goal.append(goal_weights)

            progress_bar.finish()
            display_single_matching_results(approx_actual, approx_goal)


def main():
    analyze_complete_graph()
    analyze_random_edge_graph()
    analyze_random_prob_graph()
    analyze_barabasi_graph()
    analyze_small_world_graph()
    analyze_near_complete()
    analyze_dominating()
    analyze_ring_graph()


if __name__ == "__main__":
    main()
