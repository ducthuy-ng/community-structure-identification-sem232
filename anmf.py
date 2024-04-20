import numpy
import numpy.typing as npt


def detect_community_structure_for_asymmetric_adjacent_matrix(
    adjacent_matrix_X: npt.NDArray,
    expected_community_num_K=2,
    seed=42,
    max_iter=10,
    tolerance_eps=1e-4,
):
    """This is the implementation of iANMF

    Args:
        adjacent_matrix_X (numpy.array): The adjacency matrix, supports **directed graph**
    """
    vertex_num_N = adjacent_matrix_X.shape[0]

    # Step 1
    communicator_indicator_H, community_internal_strength_S = (
        _init_nonnegative_communicator_indicator_H_and_community_internal_strength_S(
            vertex_num_N, expected_community_num_K, seed
        )
    )

    _calculate_next_communicator_indicator_H(
        adjacent_matrix_X, communicator_indicator_H, community_internal_strength_S
    )

    iter_count = 0
    while (
        iter_count < max_iter
        and calculate_i_divergence_Dx(
            adjacent_matrix_X,
            (
                communicator_indicator_H
                @ community_internal_strength_S
                @ communicator_indicator_H.T
            ),
        )
        > tolerance_eps
    ):
        new_communicator_indicator_H = _calculate_next_communicator_indicator_H(
            adjacent_matrix_X, communicator_indicator_H, community_internal_strength_S
        )

        communicator_indicator_H = new_communicator_indicator_H


def _init_nonnegative_communicator_indicator_H_and_community_internal_strength_S(
    vertex_num_N, expected_community_count_K, seed: int | None = None
):
    numpy.random.seed(seed)
    communicator_indicator_H = numpy.random.rand(
        vertex_num_N, expected_community_count_K
    )
    community_internal_strength_S = numpy.random.rand(
        expected_community_count_K, expected_community_count_K
    )

    return communicator_indicator_H, community_internal_strength_S


def _calculate_next_communicator_indicator_H(
    adjacency_matrix_X: npt.NDArray[numpy.float64],
    communicator_indicator_H: npt.NDArray[numpy.float64],
    community_internal_strength_S: npt.NDArray[numpy.float64],
):
    caching_HS = communicator_indicator_H @ community_internal_strength_S
    caching_SH_T = community_internal_strength_S @ communicator_indicator_H.T
    caching_HSH_T = (
        communicator_indicator_H
        @ community_internal_strength_S
        @ communicator_indicator_H.T
    )

    vectorized_addend_1 = (adjacency_matrix_X / caching_HSH_T) @ caching_HS
    vectorized_addend_2 = (adjacency_matrix_X / caching_HSH_T).T @ caching_SH_T.T

    extracted_dividend = numpy.sum(caching_HS + caching_SH_T.T, axis=0)
    return (
        communicator_indicator_H
        * (vectorized_addend_1 + vectorized_addend_2)
        / extracted_dividend
    )


def calculate_i_divergence_Dx(
    matrix_a: npt.NDArray[numpy.float64], matrix_b: npt.NDArray[numpy.float64]
):
    """_summary_

    Args:
        matrix_a (numpy.ndarray): _description_
        matrix_b (numpy.ndarray): _description_

    Returns:
        numpy.ndarray: A 1x1 array represented the calculated value
    """
    vectorize_inner_formula = (
        matrix_a * numpy.log10(matrix_a / matrix_b) - matrix_a + matrix_b
    )
    return numpy.sum(vectorize_inner_formula)
