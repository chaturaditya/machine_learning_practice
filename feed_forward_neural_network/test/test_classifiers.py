import numpy as np
import unittest

from assignment2_advanced.src import classifiers
from assignment2_advanced.test.gradient_check import grad_check_sparse
from assignment2_advanced.test.utils import rel_error, print_test

class TestSoftmax(unittest.TestCase):
    def test_output_softmax(self):
        print_test("Test output softmax")
        np.random.seed(395)
        X = np.random.randn(3073, 10)
        y = np.random.uniform(0, 10, 3073).astype(np.int16)
        expected_loss = 2.7536201727859835 
        loss, _ = classifiers.softmax(X, y)
        e = rel_error(loss, expected_loss)
        print("Relative error", e)
        self.assertTrue(e <= 5e-12)

    def test_derivative_softmax(self):
        print_test("Test gradients softmax")
        np.random.seed(395)
        X = np.random.randn(5, 10)
        y = np.random.uniform(0, 10, 5).astype(np.int16)
        expected_grads = np.asarray(
            [[ 0.00322816,  0.01188412,  0.00185757,  0.04150573, -0.19303544,
             0.00260126,  0.08516417,  0.02764608,  0.00912691,  0.01002143],
           [-0.19497483,  0.00531375,  0.04302849,  0.03511828,  0.00303517,
             0.04190564,  0.0038523 ,  0.01478627,  0.02427056,  0.02366437],
           [-0.18935209,  0.00978672,  0.00897249,  0.02791175,  0.04489496,
             0.00676774,  0.0171222 ,  0.00539931,  0.0294198 ,  0.03907712],
           [ 0.02925289, -0.19129923,  0.00521769,  0.00132343,  0.01317178,
             0.03008875,  0.00245856,  0.00113738,  0.01814663,  0.09050212],
           [ 0.01018762,  0.0232368 ,  0.0049424 ,  0.01980616,  0.01467162,
             0.00463897,  0.03473582,  0.007073  ,  0.03897701, -0.15826939]])
        _, grads = classifiers.softmax(X, y)
        e = rel_error(expected_grads, grads)
        print("Relative error", e)
        self.assertTrue(e <= 1e-05)

if __name__ == "__main__":
    unittest.main()
