import numpy
import numpy.typing as npt
import scipy
import scipy.sparse


class VectorizedCommunityIndicatorHCalculator:
    @classmethod
    def calculate(
        cls,
        adjacency_matrix_X: npt.NDArray[numpy.float64] | scipy.sparse.csr_matrix,
        community_indicator_H: npt.NDArray[numpy.float64],
    ) -> npt.NDArray[numpy.float64]:
        """Vectorized version, helps a lot with performance

        Args:
            adjacency_matrix_X (npt.NDArray[numpy.float64]): _description_
            community_indicator_H (npt.NDArray[numpy.float64]): _description_

        Returns:
            npt.NDArray[numpy.float64]: _description_
        """
        caching_HH_T = community_indicator_H @ community_indicator_H.T

        vectorized_sum = numpy.divide(adjacency_matrix_X, caching_HH_T, where=caching_HH_T != 0) @ community_indicator_H

        extracted_dividend = numpy.sum(community_indicator_H, axis=0)
        return community_indicator_H * vectorized_sum / extracted_dividend
