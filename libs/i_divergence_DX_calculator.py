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
        vectorize_inner_formula = matrix_a * numpy.log(matrix_a / matrix_b) - matrix_a + matrix_b

        # Making sure 0*inf = 0, not NaN
        inner_formula_with_zeros_filled = numpy.nan_to_num(vectorize_inner_formula, nan=0)
        return numpy.sum(inner_formula_with_zeros_filled)
