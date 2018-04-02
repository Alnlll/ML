import sys
import os

sys.path.append("%s\\.." % os.getcwd())

import numpy as np
from arsenal.data_process.load_data import DataLoader
from arsenal.data_process.plot_data import Ploter
from arsenal.data_process.normalize_data import Normalizer

class Pca(object):
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_loader = DataLoader()
        self.data_ploter = Ploter()
        self.data_normalizer = Normalizer()

    def get_covar(self, data):
        return np.dot(data.T, data) / data.shape[0]

    def svd(self, data):
        return np.linalg.svd(data)

    def get_u_reduce(self, U, k):
        return U[:, 0:k]

    def compress(self, data, u_reduce):
        return np.dot(data, u_reduce)

    def recover(self, data, u_reduce):
        return np.dot(data, u_reduce.T)


if '__main__' == __name__:
    test = Pca('./data/ex7data1.mat')
    data = test.data_loader.load(test.data_path)['X']
    test.data_ploter.plot(data[:,0], data[:,1])
    
    mu, sigma, data = test.data_normalizer.normalize(data)
    test.data_ploter.plot(data[:,0], data[:,1])

    con = test.get_covar(data)

    U, S, V = test.svd(con)

    print("mu: ", mu)
    print('sigma: ', sigma)
    print('norm: ', data)
    print("U: ", U)
    print("S: ", S)

    print("U(:,1) = ", U[:,0])

    u_reduce = test.get_u_reduce(U, 1)
    data = test.compress(data, u_reduce)
    print('Projection of the first example:\n', data[0]);
    print('\n(this value should be about 1.481274)\n\n');

    data = test.recover(data, u_reduce)
    test.data_ploter.plot(data[:,0], data[:,1])
    print('Approximation of the first example:\n', data[0, 0], data[0, 1]);
    print('\n(this value should be about  -1.047419 -1.047419)\n\n');
