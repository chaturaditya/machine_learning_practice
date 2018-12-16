import numpy as np


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def print_test(s):
    print("\n==============================================")
    print(s)
    print("==============================================")
