import numpy
import numpy.typing as npt

from libs.community_indicator_calculator.vectorized import VectorizedCommunityIndicatorHCalculator
from libs.i_divergence_DX_calculator import IDivergenceDxCalculator


def detect_community_structure_for_symmetric_adjacent_matrix(
    adjacent_matrix_X: npt.NDArray,
    expected_community_num_K=2,
    max_iter=500,
    tolerance_eps=1e-2,
    community_membership_scale_alpha=0.9,
    seed=42,
):
    """This is the implementation of iSNMF.

    Whichever of `max_iter` and `tolerance_eps` comes first will stop the main iteration.

    Args:
        adjacent_matrix_X (npt.NDArray): The adjacency matrix, only support undirected graph.
        expected_community_num_K (int, optional): How many community to expect. Defaults to 2.
        max_iter (_type_, optional):
            How much iterations should we perform. Defaults to 500.
        tolerance_eps (_type_, optional): How much should I-Divergence score change to be consider insignificant.
            Defaults to 1e-4.
        community_membership_scale_alpha (float, optional):
            How low can the community indicator be to be consider belonging to a community.
            Defaults to 0.9.
        seed (int, optional): Random seed for community indicator H. Defaults to 42.

    Returns:
        _type_: _description_
    """
    vertex_num_N = adjacent_matrix_X.shape[0]

    # Step 1
    numpy.random.seed(seed)
    community_indicator_H = numpy.random.rand(vertex_num_N, expected_community_num_K)

    # Modification:
    # Experiments shows that Dx threshold varies a lot.
    # Therefore, we will stop if iter does not decrease Dx, instead.
    iter_count = 0
    prev_Dx = numpy.inf
    current_Dx = None
    while True:
        current_Dx = IDivergenceDxCalculator.calculate(
            adjacent_matrix_X,
            community_indicator_H @ community_indicator_H.T,
        )

        if iter_count >= max_iter or numpy.abs(current_Dx - prev_Dx) < tolerance_eps:
            break
        prev_Dx = current_Dx

        community_indicator_H = VectorizedCommunityIndicatorHCalculator.calculate(
            adjacent_matrix_X, community_indicator_H
        )

        iter_count += 1

    # Step 3
    communities_C = [set() for i in range(expected_community_num_K)]
    normalized_community_indicator_P = normalize_community_indicator_H(community_indicator_H)
    max_community_indicator_per_node = normalized_community_indicator_P.max(axis=1, keepdims=True)

    node_community_pairs = numpy.argwhere(
        normalized_community_indicator_P > community_membership_scale_alpha * max_community_indicator_per_node
    )
    for node_community_pair in numpy.nditer(node_community_pairs, flags=["external_loop"], order="C"):
        node, community = node_community_pair[0], node_community_pair[1]
        communities_C[community].add(node)

    return communities_C


def normalize_community_indicator_H(community_indicator_H: npt.NDArray[numpy.float64]) -> npt.NDArray[numpy.float64]:
    return community_indicator_H / numpy.linalg.norm(community_indicator_H, axis=1, keepdims=True)
