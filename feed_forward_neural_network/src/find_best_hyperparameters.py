import numpy as np
import matplotlib.pyplot as plt
import os
from bayes_opt import BayesianOptimization

from assignment2_advanced.src.fcnet import FullyConnectedNet
from assignment2_advanced.src.utils.solver import Solver
from assignment2_advanced.src.utils.load_fer import load_fer_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################

# settings
num_train_images = 28708 # out of 28708
num_test_images = 3588 # out of 3588

# load images
PATH = os.getcwd()
print('PATH:', PATH)
img_folder = PATH + '/datasets/FER2013'
X_train, y_train, X_val, y_val = load_fer_data(img_folder, num_train_images, num_test_images)
data = {
  'X_train': X_train, 'y_train': y_train,
  'X_val': X_val, 'y_val': y_val,
}

# make grayscale
data['X_train'] = data['X_train'].mean(axis=3)
data['X_val'] = data['X_val'].mean(axis=3)

def model_accuracy(hidden_dims_1, hidden_dims_2, learning_rate_exp, momentum, batch_size, reg=0, dropout=0):
    hidden_dims_1 = int(hidden_dims_1)
    hidden_dims_2 = int(hidden_dims_2)
    batch_size = int(batch_size)

    hidden_dims = [hidden_dims_1, hidden_dims_2]

    # Create net and solver and train
    model = FullyConnectedNet(hidden_dims=hidden_dims,
                              input_dim=np.prod(data['X_train'].shape[1:]),
                              num_classes=7,
                              dropout=0,
                              reg=reg,
                              weight_scale=1e-3)

    solver = Solver(model, data,
                    update_rule='sgd_momentum', # ['sgd', 'sgd_momentum']
                    optim_config={
                      'learning_rate': 10 ** learning_rate_exp,
                      'momentum': momentum
                    },
                    lr_decay=0.95,
                    num_epochs=40,
                    batch_size=batch_size,
                    print_every=100,
                    verbose=False)
    solver.train()

    # calculate accuracy
    return solver.check_accuracy(data['X_val'], data['y_val'], batch_size=batch_size)

bo = BayesianOptimization(model_accuracy,
                          {'hidden_dims_1': (20, 1000),
                           'hidden_dims_2': (20, 1000),
                           'reg': (0, .5),
                           'learning_rate_exp': (-5, -3),
                           'momentum': (0.0, 1.0),
                           'batch_size': (50, 200)})

bo.maximize(init_points=5, n_iter=100, kappa=2)
print(bo.res['max'])


##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
