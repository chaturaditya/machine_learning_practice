# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

# NB this is tested on python 2.7. Watch out for integer division

import numpy as np

A=np.array([[8, -2], [-2, 8]])
B=np.array([1, 1])


def grad_f1(x):
    """
    4 marks

    :param x: input array with shape (2, )
    :return: the gradient of f1, with shape (2, )
    """

    grad = np.dot(A,x)-B
    return grad

def grad_f2(x):
    """
    6 marks

    :param x: input array with shape (2, )
    :return: the gradient of f2, with shape (2, )
    """
    
    grad1 = 2*(x[0]-1)*np.cos(np.power(x[0]-1,2)+np.power(x[1],2))+6*x[0]-2-2*x[1]
    grad2 = 2*(x[1])*np.cos(np.power(x[0]-1,2)+np.power(x[1],2))+6*x[1]+6-2*x[0]
    
    return np.array([grad1, grad2])

def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """

    grad3 = (2*(x[0]-1)*np.exp(-x[0]**2+2*x[0]-x[1]**2-1) + 0.2*x[0]/(x[0]**2+x[1]**2+0.01) + np.exp(-3*x[0]**2 + 2*x[0]*(x[1]+1) -3 *(x[1]+1)**2)*(6*x[0]-2*(x[1]+1)))
    grad4 = (2*x[1]*np.exp(-x[0]**2+2*x[0]-x[1]**2-1)+0.2*x[1]/(x[0]**2+x[1]**2+0.01) + np.exp(-3*x[0]**2+2*x[0]*(x[1]+1)-3*(x[1]+1)**2)*(-2*x[0]+6*x[1]+6))
  
  
    return np.array([grad3, grad4])
