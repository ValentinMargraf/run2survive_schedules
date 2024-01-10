import numpy.testing as npt
import numpy as np
from survival_tests.save_results import save_survival_curves, read_survival_curves

def test_save_and_read_survival_curves():
    '''
    This test checks if the saving of numpy array of many dimensions works.
    '''
    input = np.ones((100, 20))
    filename = "test"
    save_survival_curves(input, filename)
    result = read_survival_curves(filename)
    npt.assert_array_almost_equal(input, np.array(result))


def test_save_and_read_survival_curves_2():
    '''
    This tests checks if hte saving of a normal list works as expected.
    '''
    input = list(range(100))
    filename = "test2"
    save_survival_curves(input, filename)
    result = read_survival_curves(filename)
    npt.assert_array_almost_equal(input, np.array(result))

