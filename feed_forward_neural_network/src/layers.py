import numpy as np


def linear_forward(X, W, b):
    """
    Computes the forward pass for a linear (fully-connected) layer.

    Args:
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: Anumpy array of shape (K, M) of weights
    - b: A numpy array of shape (M, ) of biases

    Returns:
    - out: linear transformation to the incoming data


    """

    #X.shape[0] - N
    X = np.reshape(X,[X.shape[0], np.prod(X.shape[1:])])
    out = np.dot(X,W) + b

    return out


def linear_backward(dout, X, W, b):
    """
    Computes the backward pass for a linear (fully-connected) layer.

    Args:
    - dout: Upstream derivative, of shape (N, M)
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: Anumpy array of shape (K, M) of weights
    - b: A numpy array of shape (M, ) of biases

    Returns (as tuple):
    - dX: A numpy array of shape (N, d1, ..., d_k), gradient with respect to x
    - dW: A numpy array of shape (D, M), gradient with respect to W
    - db: A nump array of shape (M,), gradient with respect to b
    """

    dX = np.dot(dout, W.transpose()).reshape(X.shape)
    dW = np.dot(X.reshape((X.shape[0], -1)).transpose(), dout)
    db = np.sum(dout, axis=0)

    return dX, dW, db


def relu_forward(X):
    """
    Computes the forward pass for rectified linear unit (ReLU) layer.
    Args:
    - X: Input, an numpy array of any shape
    Returns:
    - out: An numpy array, same shape as X
    """

    out = X.copy()  # Must use copy in numpy to avoid pass by reference.
    out[out < 0] = 0

    return out


def relu_backward(dout, X):
    """
    Computes the backward pass for rectified linear unit (ReLU) layer.
    Args:
    - dout: Upstream derivative, an numpy array of any shape
    - X: Input, an numpy array with the same shape as dout
    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    dX = dout * (X >= 0)

    return dX


def dropout_forward(X, p=0.5, train=True, seed=False):
    """
    Compute f
    Args:
    - X: Input data, a numpy array of any shape.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns (as a tuple):
    - out: Output of dropout applied to X, same shape as X.
    - mask: In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    if seed:
        np.random.seed(seed)
    q = 1-p
    if train == True:
        mask = 1/q*np.random.binomial(1,p=q, size = X.shape)
        out = np.multiply(mask,X)
    else:
        mask = None
        out = X

    return out, mask


def dropout_backward(dout, mask, p=0.5, train=True):
    """
    Compute the backward pass for dropout
    Args:
    - dout: Upstream derivative, a numpy array of any shape
    - mask: In training mode, mask is the dropout mask that was used to
      multiply the input; in test mode, mask is None.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    if train == True:
        dX = mask*dout
    else:
        dX = dout
    return dX
