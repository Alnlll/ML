import numpy as np
from arsenal.common.basic_func import basic_func
from arsenal.optimizer.grad_descent import GradDescent

class NueralNetwork(object):
    """docstring for NueralNetwork"""
    def __init__(self, map, optimizer=GradDescent):
        self.map = map
        self.L = len(map.keys()) - 1
        self.optimizer = optimizer()

    def initialize_params(self, seed=1):
        np.random.seed(seed)

        for l in range(1, self.L+1):
            params = self.map['L'+str(l)]

            # Get number of neuron
            n_prev = self.map['L'+str(l-1)]['n_neuron']
            n = params['n_neuron']

            # Shape of W[l] should be (n[l], n[l-1]), and W[l] should initialized randomly to break sync.
            params['W'] = np.random.randn(n, n_prev) / np.sqrt(n_prev)
            # Shape of b[l] should be (n[l], 1).
            params['b'] = np.zeros((n, 1))

    def loss(self, Y_hat, Y):
        m = Y.shape[1]

        logpart = -Y*np.log(Y_hat) - (1-Y)*np.log(1-Y_hat)
        loss = np.sum(logpart) / m

        loss = np.squeeze(loss)

        return loss

    def forward_propagation(self, X, start = 1): 
        assert(1 <= start <= self.L)

        self.map['L0']['a'] = X
        
        for l in range(start, self.L+1):
            params = self.map['L'+str(l)]

            a_prev = self.map['L'+str(l-1)]['a']
            params['z'] = np.dot(params['W'], a_prev) + params['b']
            params['a'] = basic_func(params['activation'], params['z'])

        aL = self.map['L'+str(self.L)]['a']

        return aL

    # The regularization is just L2 
    def backward_propagation(self, daL, end = 1, reg_flag=False, lambd=0.1):
        assert(1 <= end <= self.L)

        for l in range(self.L, end-1, -1):
            params = self.map['L'+str(l)]

            # Get da
            if l == self.L:
                da = daL
            else:
                da = np.dot(self.map['L'+str(l+1)]['W'].T, self.map['L'+str(l+1)]['dz'])

            # Get dz
            da_z = basic_func('d'+params['activation'], params['z'])
            dz = da * da_z
            params['dz'] = dz

            # Get dW, db
            a_prev = self.map['L'+str(l-1)]['a']
            dz_w = a_prev.T

            # Calculate regularization item
            if reg_flag:
                regu_item_W = lambd * params['dW'];
                regu_item_b = lambd * params['db'];
            if not reg_flag:
                regu_item_W = 0;
                regu_item_b = 0;

            params['dW'] = np.dot(dz, dz_w) + regu_item_W
            params['db'] = np.sum(dz, axis=1, keepdims=True) + regu_item_b

    def update_params(self, rate=0.1):
        for l in range(1, self.L+1):
            params = self.map['L'+str(l)]

            params['W'] = self.optimizer.descent(params['dW'], params['W'], rate=rate)
            params['b'] = self.optimizer.descent(params['db'], params['b'], rate=rate)

    def predict(self, X, threshold=0.5):

        p = self.forward_propagation(X)

        p[p >= threshold] = 1
        p[p < threshold] = 0

        return p
