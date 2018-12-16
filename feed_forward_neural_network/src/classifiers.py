import numpy as np

def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None

    N = logits.shape[0]
    exps = softmax_classifier(logits,N)
    assert(np.allclose(np.sum(exps, axis=1), np.ones(N)))
    exp_sums = np.array([exps[i][y[i]] for i in range(N)])
    exp_sums[exp_sums == 0] = 1e-40 # handles log of 0 warning
    loss = 1/N*np.sum(-np.log(exp_sums))
    m = y.shape[0]
    # https://deepnotes.io/softmax-crossentropy
    dlogits = exps.copy()
    dlogits[range(m), y] -= 1
    dlogits = dlogits / m

    return loss, dlogits


def softmax_classifier(x,N):
    return np.array([np.exp(x[i] - np.max(x[i])) / np.sum(np.exp(x[i] - np.max(x[i])))
              for i in range(N)])
