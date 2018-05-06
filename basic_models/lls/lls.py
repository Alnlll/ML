import sys
import os

sys.path.append("%s\\.." % os.getcwd())

import numpy as np
from arsenal.data_process.plot_data import Ploter
from arsenal.optimizer.grad_descent import GradDescent

class LLS(object):
    """Linear least square"""
    def __init__(self):
        self.ploter = Ploter()
    def lls_func(self, X, A, b):
        return (1/2.) * (np.linalg.norm(np.dot(A,X) - b)**2)
    def lls_grad(self, X, A, b):
        return np.dot(A.T, (np.dot(A,X) - b))


if '__main__' == __name__:

    
    epsilon = 1e-7;
    #A = np.array(([1,2,3,4,5],[3,3,1,2,5],[1,4,3,2,5],[1,8,3,10,5],[1,3,1,4,9],[1,0.5,3.2,4,5]))
    A = np.random.randn(6,5)*3
    x = np.random.randn(A.shape[1],1)
    b = np.random.randn(A.shape[0],1)

    count = 0
    values = []

    test = LLS()
    optimizer = GradDescent()

    while (np.linalg.norm(test.lls_grad(x, A, b)) > epsilon) and (count < 5000):
        if 0 == (count % 10):
            values.append(test.lls_func(x, A, b))
            print("Cost(%d): %f" % (count, values[-1]))
        x = optimizer.descent(test.lls_grad(x, A, b), x)
        count += 1

    index = np.arange(0,count,10)
    test.ploter.plot(index, np.array((values)), set_str = 'r-')
