import numpy as np
import unittest

'''
#Anya - otherwise does not run
import sys
sys.path.append('/Users/a123/Documents/Imperial/Year4/Courses/ML_395/CW/395_machine_learning/assignment2_advanced/src/')
import layers
sys.path.append('/Users/a123/Documents/Imperial/Year4/Courses/ML_395/CW/395_machine_learning/assignment2_advanced/test/')
from gradient_check import eval_numerical_gradient_array
sys.path.append('/Users/a123/Documents/Imperial/Year4/Courses/ML_395/CW/395_machine_learning/assignment2_advanced/test/')
from utils import rel_error, print_test'''

from assignment2_advanced.src import layers
from assignment2_advanced.test.gradient_check import eval_numerical_gradient_array
from assignment2_advanced.test.utils import rel_error, print_test


class TestLinearLayer(unittest.TestCase):

    def test_output_linear_forward(self):
        print_test("Testing linear forward function:")
        n_X = 10
        X_shape = [4, 2, 5]
        n_out = 3
        n_input = n_X * np.prod(X_shape)
        n_weights = np.prod(X_shape) * n_out
        X = np.linspace(-0.2, 0.5, num=n_input).reshape([n_X] + X_shape)
        W = np.linspace(-0.4, 0.2, num=n_weights).reshape(
            np.prod(X_shape), n_out)
        b = np.linspace(0.5, 1, num=n_out)
        out = layers.linear_forward(X, W, b)
        correct_out = np.asarray([[1.33803627, 1.55459973, 1.7711632],
                                  [1.04318148, 1.27389798, 1.50461448],
                                  [0.7483267 , 0.99319623, 1.23806575],
                                  [0.45347192, 0.71249447, 0.97151703],
                                  [0.15861713, 0.43179272, 0.7049683],
                                  [-0.13623765, 0.15109096, 0.43841958],
                                  [-0.43109244, -0.12961079, 0.17187085],
                                  [-0.72594722, -0.41031255, -0.09467787],
                                  [-1.02080201, -0.6910143 , -0.3612266 ],
                                  [-1.31565679, -0.97171605, -0.62777532]])
        e = np.max(np.abs(correct_out - out))
        e = rel_error(out, correct_out)
        print("Relative difference", e)
        self.assertTrue(e <= 5e-08)

    def test_output_linear_backward(self):
        print_test("Testing linear backward function:")
        np.random.seed(395)
        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        dx_num = eval_numerical_gradient_array(
            lambda x: layers.linear_forward(x, w, b), x, dout)
        dw_num = eval_numerical_gradient_array(
            lambda w: layers.linear_forward(x, w, b), w, dout)
        db_num = eval_numerical_gradient_array(
            lambda b: layers.linear_forward(x, w, b), b, dout)

        dx, dw, db = layers.linear_backward(dout, x, w, b)

        dX_e = rel_error(dx_num, dx)
        dW_e = rel_error(dw_num, dw)
        db_e = rel_error(db_num, db)
        print('dX relative error: ', dX_e)
        print('dW relative error: ', dW_e)
        print('db realtive error: ', db_e)
        self.assertTrue(dX_e <= 5e-10)
        self.assertTrue(dW_e <= 5e-10)
        self.assertTrue(db_e <= 5e-10)


class TestReLULayer(unittest.TestCase):

    def test_output_relu_forward(self):
        print_test("Testing relu forward function:")
        x = np.linspace(-0.7, 0.5, num=20).reshape(5, 4)
        out = layers.relu_forward(x)
        correct_out = np.array([[0., 0., 0., 0.],
                                [0., 0., 0., 0.],
                                [0., 0., 0., 0.],
                                [0.05789474, 0.12105263, 0.18421053, 0.24736842],
                                [0.31052632, 0.37368421, 0.43684211, 0.5]])
        e = rel_error(out, correct_out)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-08)

    def test_output_relu_backward(self):
        print_test("Testing relu backward function:")
        np.random.seed(395)
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)
        dx_num = eval_numerical_gradient_array(
            lambda x: layers.relu_forward(x), x, dout)
        dx = layers.relu_backward(dout, x)
        dx_e = rel_error(dx, dx_num)
        print('dX relative difference:', dx_e)
        self.assertTrue(dx_e <= 1e-11)


class TestDropoutLayer(unittest.TestCase):

    def test_output_dropout_forward(self):
        print_test("Testing dropout forward function:")
        seed = 395
        np.random.seed(seed)
        p = 0.7
        x = np.linspace(-0.7, 0.5, num=10).reshape(2, 5)
        out_train, _ = layers.dropout_forward(x, p=p, seed=seed, train=True)

        out_test, _ = layers.dropout_forward(x, p=p, seed=seed, train=False)
        correct_out = np.asarray([[-0., -0., -0., -1., -0.],
                                  [-0.,  0.,  0.,  0.,  0.]])
        e_train = rel_error(out_train, correct_out)
        e_test = rel_error(out_test, x)
        print("Relative difference train:", e_train)
        print("Relative difference test:", e_test)
        self.assertTrue(e_train < 1e-12)
        self.assertTrue(e_test < 1e-12)


    def test_output_relu_backward(self):
        print_test("Testing dropout backward function:")
        seed = 395
        np.random.seed(seed)
        x = np.random.randn(16, 16) + 8
        dout = np.random.randn(*x.shape)
        p = 0.7
        # Test for train
        dout, mask = layers.dropout_forward(x, train=True, p=p, seed=seed)
        dx = layers.dropout_backward(dout, mask, p=p, train=True)
        dx_num = eval_numerical_gradient_array(
            lambda xx: layers.dropout_forward(
                xx, p=0.7, train=True, seed=seed)[0], x, dout)
        e_train = rel_error(dx, dx_num)
        print('dx train relative error: ', e_train)
        self.assertTrue(e_train <= 5e-11)
        # Test for test
        dx_test = layers.dropout_backward(dout, mask, train=False, p=p)
        e_test = rel_error(dout, dx_test)
        print('dx test relative error: ', e_test)
        self.assertTrue(e_test <= 1e-12)


if __name__ == "__main__":
    unittest.main()
