import numpy

from libs.snmf import detect_community_structure_for_symmetric_adjacent_matrix


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

    community_C = detect_community_structure_for_symmetric_adjacent_matrix(test_adjacency_matrix)

    community_result = [{0, 1, 2, 3}, {3, 4, 5, 6, 7}]
    for community in community_C:
        assert community in community_result
