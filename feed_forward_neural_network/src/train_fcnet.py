import numpy as np
import matplotlib.pyplot as plt

from assignment2_advanced.src.fcnet import FullyConnectedNet
from assignment2_advanced.src.utils.solver import Solver
from assignment2_advanced.src.utils.data_utils import get_CIFAR10_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################

# Get data and limit to 50 samples
#data = get_CIFAR10_data(10000, 1000, 1000, True)
data = get_CIFAR10_data()

# Create net and solver and train
model = FullyConnectedNet(hidden_dims=[60, 75],
                          input_dim=32*32*3,
                          num_classes=10,
                          dropout=0,
                          reg=0.2,
                          weight_scale=1e-2)

solver = Solver(model, data,
                update_rule='sgd_momentum', # ['sgd', 'sgd_momentum']
                optim_config={
                  'learning_rate': 2e-3,
                },
                lr_decay=0.95,
                num_epochs=40,
                batch_size=100,
                print_every=100)
solver.train()

# Check accuracy
acc = solver.check_accuracy(data['X_test'], data['y_test'])
print('Test set accuracy:', acc)

# Plot errors
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
