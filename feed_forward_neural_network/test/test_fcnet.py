import numpy as np

from assignment2_advanced.src import fcnet
from assignment2_advanced.test.utils import rel_error
from assignment2_advanced.test.gradient_check import eval_numerical_gradient

if __name__ == "__main__":
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print('Running check with reg = ', reg)
        model = fcnet.FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                                        reg=reg, dtype=np.float64)
        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)
        print(model.params.keys())
        print(grads.keys())
        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name],
                                               verbose=False, h=1e-5)
            print('{name} relative error: {rel:.2e}'.format(
                name=name, rel=rel_error(grad_num, grads[name])))
