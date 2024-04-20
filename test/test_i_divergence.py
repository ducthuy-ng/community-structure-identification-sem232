import numpy
from anmf import calculate_i_divergence_Dx


def test_calculate_i_divergence_Dx():
    matrix_A = numpy.array([[1.0, 2.0]])
    matrix_B = numpy.array([[3.0, 4.0]])

    result = calculate_i_divergence_Dx(matrix_A, matrix_B)

    # Use isclose for tolerance
    assert numpy.isclose(result, numpy.array([[2.920818754]]))
