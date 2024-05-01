import numpy
import numpy.typing as npt


class IDivergenceDxCalculator:
    @classmethod
    def calculate(cls, matrix_a: npt.NDArray[numpy.float64], matrix_b: npt.NDArray[numpy.float64]):
        """Calculate I-divergence of 2 matrices (also known as Kullback-Leibner divergence).
        Notation: D(A||B) - the I-divergence of matrix A from matrix B.

        This function has 2 properties that make it a good candidate for scoring and distance measurement:
        1. D(A||B) >= 0
        2. D(A||B) iff A = B

        # Note:
        - This operator is not interchangable:      D(A||B) != D(B||A)
        - It must be the natural log (Napierian logarithm - Log nepe in Vietnamese): x * ln(x) >= x - 1

        Args:
            matrix_a (numpy.ndarray): _description_
            matrix_b (numpy.ndarray): _description_

        Returns:
            numpy.ndarray: _description_
        """
        # 0 / 0 = 0 (assumption)
        matrix_a_times_log = numpy.where(
            matrix_a != 0,
            matrix_a * numpy.log(numpy.divide(matrix_a, matrix_b, where=matrix_b != 0)),
            matrix_a,
        )
        vectorize_inner_formula = matrix_a_times_log - matrix_a + matrix_b

        return numpy.sum(vectorize_inner_formula)
