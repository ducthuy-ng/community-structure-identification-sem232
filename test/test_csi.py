import numpy
from anmf import (
    calculate_i_divergence_Dx,
    detect_community_structure_for_asymmetric_adjacent_matrix,
)


def test_calculate_i_divergence_Dx():
    matrix_A = numpy.array([[1.0, 2.0]])
    matrix_B = numpy.array([[3.0, 4.0]])

    result = calculate_i_divergence_Dx(matrix_A, matrix_B)

    # Use isclose for tolerance
    assert numpy.isclose(result, numpy.array([[2.920818754]]))


def test_detect_community_structure_for_asymmetric_adjacent_matrix():
    test_adjacency_matrix = numpy.array(
        [
            [0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 1, 1, 0],
        ]
    )

    detect_community_structure_for_asymmetric_adjacent_matrix(test_adjacency_matrix)
    ...
