import numpy as np

from assignment2_advanced.src.classifiers import softmax, softmax_classifier
from assignment2_advanced.src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)



def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    W = None
    b = None
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    #initialise weight matrix:
    W = np.random.normal(0, weight_scale, (n_in, n_out))

    #initialize the biased
    b = np.zeros(n_out)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return W, b



class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10,
                 dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: A list giving the size of the input
        - num_classes: Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.

        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the Xavier
        initialisation (see manual).
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        dim_list = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            in_dim = dim_list[i]
            out_dim = dim_list[i+1]
            #print('W' + str(i), 'has shape', in_dim, out_dim)
            W_init, b_init = random_init(in_dim, out_dim, weight_scale=5e-2, dtype=np.float32)
            self.params['W'+str(i)] = W_init
            self.params['b'+str(i)] = b_init

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.


        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"train": True, "p": dropout}
            if seed is not None:
                self.dropout_params["seed"] = seed
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        # [linear - relu - (dropout)] x (N - 1) - linear - softmax

        X = np.reshape(X,[X.shape[0], np.prod(X.shape[1:])])
        out = np.copy(X)

        for i in range(self.num_layers-1):
            W = self.params['W'+str(i)]
            b = self.params['b'+str(i)]
            linear_cache[str(i)] = out
            out = linear_forward(out, W, b)
            relu_cache[str(i)] = out
            out = relu_forward(out)
            # check if using dropout and not in test mode
            if self.use_dropout and y is not None:
                out, mask = dropout_forward(out, **self.dropout_params)
                dropout_cache[str(i)] = mask

        final_num = str(self.num_layers - 1)
        linear_cache[final_num] = out
        scores = linear_forward(out, self.params['W'+final_num], self.params['b'+final_num])

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, dict()

        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        # get errors and regularised loss
        loss, dlogits = softmax(scores, y)
        loss += 0.5 * self.reg * np.sum(self.params['W'+final_num] ** 2)

        # backprop final linear and store gradients
        final_num = str(self.num_layers - 1)
        dX, dW, db = linear_backward(dlogits, linear_cache[final_num], self.params['W'+final_num], self.params['b'+final_num])
        grads['W'+final_num] = dW + self.reg * self.params['W'+final_num]
        grads['b'+final_num] = db

        # backprop all layers
        for i in range(self.num_layers-1 - 1, -1, -1): # count backwards
            W = self.params['W'+str(i)]
            b = self.params['b'+str(i)]
            if self.use_dropout:
                dX = dropout_backward(dX, dropout_cache[str(i)], **self.dropout_params)
            dX = relu_backward(dX, relu_cache[str(i)])
            dX, dW, db = linear_backward(dX, linear_cache[str(i)], W, b)

            # store weight gradients with shrinkage
            grads['W'+str(i)] = dW + self.reg * self.params['W'+str(i)]
            grads['b'+str(i)] = db

            # add regularisation to loss for each layer.
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i)] ** 2)

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        return loss, grads
