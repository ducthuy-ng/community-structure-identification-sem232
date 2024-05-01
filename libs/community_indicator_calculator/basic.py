import numba
import numpy
import numpy.typing as npt
import scipy
import scipy.sparse


class BasicCommunityIndicatorHCalculator:
    @classmethod
    def calculate(
        cls,
        adjacency_matrix_X: npt.NDArray[numpy.float64] | scipy.sparse.csr_array,
        community_indicator_H: npt.NDArray[numpy.float64],
    ) -> npt.NDArray[numpy.float64]:
        """Numba does not support JIT for classmethod.
        Therefore, I have to split the main logic as a private method

        Args:
            adjacency_matrix_X (npt.NDArray[numpy.float64]): _description_
            community_indicator_H (npt.NDArray[numpy.float64]): _description_

        Returns:
            npt.NDArray[numpy.float64]:
        """
        if isinstance(adjacency_matrix_X, scipy.sparse.csr_array):
            adjacency_matrix_X = adjacency_matrix_X.toarray()
        return _vectorized_calculate(adjacency_matrix_X, community_indicator_H)


@numba.jit
def _vectorized_calculate(
    adjacency_matrix_X: npt.NDArray[numpy.float64],
    community_indicator_H: npt.NDArray[numpy.float64],
) -> npt.NDArray[numpy.float64]:
    new_community_indicator_H = numpy.zeros_like(community_indicator_H)

    caching_HH_T = community_indicator_H @ community_indicator_H.T

    for row_index_a in range(new_community_indicator_H.shape[0]):
        for column_index_b in range(new_community_indicator_H.shape[1]):
            divisor_sum = 0
            for index_k in range(new_community_indicator_H.shape[0]):
                divisor_sum += (
                    community_indicator_H[index_k, column_index_b]
                    * adjacency_matrix_X[row_index_a, index_k]
                    / caching_HH_T[row_index_a, index_k]
                )

            dividend_sum = 0
            for index_t in range(new_community_indicator_H.shape[0]):
                dividend_sum += community_indicator_H[index_t, column_index_b]

            new_community_indicator_H[row_index_a, column_index_b] = community_indicator_H[
                row_index_a, column_index_b
            ] * (divisor_sum / dividend_sum)

    return new_community_indicator_H
