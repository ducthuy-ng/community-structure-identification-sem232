import networkx
import numpy
from snmf import calculate_next_community_indicator_H


def test_calculate_next_communicator_indicator_H():
    adjacency_matrix = numpy.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )

    randomize_community_indicator_H = numpy.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    next_communicator_indicator_H = calculate_next_community_indicator_H(
        adjacency_matrix, randomize_community_indicator_H
    )

    caching_HH_T = randomize_community_indicator_H @ randomize_community_indicator_H.T

    a = 0
    b = 0

    divisor = 0.0
    for k in range(adjacency_matrix.shape[0]):
        divisor += (
            randomize_community_indicator_H[k, b]
            * adjacency_matrix[a, k]
            / caching_HH_T[a, k]
        )

    dividend = 0.0
    for t in range(randomize_community_indicator_H.shape[0]):
        dividend += randomize_community_indicator_H[t, b]

    assert next_communicator_indicator_H[a, b] == randomize_community_indicator_H[
        0, 0
    ] * (divisor / dividend)


def test_stress_test_calculation():
    SEED = 42
    vertex_num_N = 1000
    edge_num = 5000
    estimated_community_K = 100

    graph: networkx.Graph = networkx.dense_gnm_random_graph(
        n=vertex_num_N, m=edge_num, seed=SEED
    )
    adjacency_matrix = networkx.adjacency_matrix(graph)

    numpy.random.seed(SEED)
    randomize_community_indicator_H = numpy.random.rand(
        vertex_num_N, estimated_community_K
    )

    next_community_indicator_H = calculate_next_community_indicator_H(
        adjacency_matrix, randomize_community_indicator_H
    )

    caching_HH_T = randomize_community_indicator_H @ randomize_community_indicator_H.T

    for N_indicator_a in range(vertex_num_N):
        for K_indicator_b in range(estimated_community_K):
            divisor = 0.0
            for k in range(vertex_num_N):
                divisor += (
                    randomize_community_indicator_H[k, K_indicator_b]
                    * adjacency_matrix[N_indicator_a, k]
                    / caching_HH_T[N_indicator_a, k]
                )

            dividend = 0.0
            for t in range(estimated_community_K):
                dividend += randomize_community_indicator_H[t, K_indicator_b]

            assert (next_community_indicator_H[N_indicator_a, K_indicator_b]) == (
                randomize_community_indicator_H[N_indicator_a, K_indicator_b]
                * (divisor / dividend)
            ), f"Failed for a, b: {N_indicator_a}, {K_indicator_b}"
