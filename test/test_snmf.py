import numpy

from snmf import (
    detect_community_structure_for_symmetric_adjacent_matrix,
)


def test_detect_community_structure_for_symmetric_adjacent_matrix():
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

    detect_community_structure_for_symmetric_adjacent_matrix(test_adjacency_matrix)


