import numpy
import numpy.typing as npt


def detect_community_structure_for_symmetric_adjacent_matrix(
    adjacent_matrix_X: npt.NDArray,
    expected_community_num_K=2,
    seed=42,
    max_iter=2_000,
    tolerance_eps=1e-4,
):
    """This is the implementation of iANMF

    Args:
        adjacent_matrix_X (numpy.array): The adjacency matrix, supports **directed graph**
    """
    vertex_num_N = adjacent_matrix_X.shape[0]

    # Step 1
    numpy.random.seed(seed)
    community_indicator_H = numpy.random.rand(vertex_num_N, expected_community_num_K)

    iter_count = 0
    while (
        iter_count < max_iter
        and calculate_i_divergence_Dx(
            adjacent_matrix_X,
            (community_indicator_H @ community_indicator_H.T),
        )
        > tolerance_eps
    ):
        community_indicator_H = calculate_next_community_indicator_H(
            adjacent_matrix_X, community_indicator_H
        )

        iter_count += 1

    pass


def calculate_next_community_indicator_H(
    adjacency_matrix_X: npt.NDArray[numpy.float64],
    community_indicator_H: npt.NDArray[numpy.float64],
):
    caching_HH_T = community_indicator_H @ community_indicator_H.T

    vectorized_sum = (adjacency_matrix_X / caching_HH_T) @ community_indicator_H

    extracted_dividend = numpy.sum(community_indicator_H, axis=0)
    return community_indicator_H * vectorized_sum / extracted_dividend


def calculate_i_divergence_Dx(
    matrix_a: npt.NDArray[numpy.float64], matrix_b: npt.NDArray[numpy.float64]
):
    """_summary_

    Args:
        matrix_a (numpy.ndarray): _description_
        matrix_b (numpy.ndarray): _description_

    Returns:
        numpy.ndarray: A 1x1 array represent        ed the calculated value
    """
    vectorize_inner_formula = (
        matrix_a * numpy.log10((matrix_a / matrix_b) + 0.0001) - matrix_a + matrix_b
    )
    return numpy.sum(vectorize_inner_formula)
