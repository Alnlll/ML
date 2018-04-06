import numpy as np

def Sigmoid(x):
    '''Calculate Sigmoid value'''
    return 1.0 / (1 + np.exp(-x))

def dSigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

def Tanh(x):
    return np.tanh(x)

def dTanh(x):
    return 1 - np.tanh(x)**2

def Relu(x):
    return np.maximum(0, x)

def dRelu(x):
    dR = np.array(x, copy=True)
    dR[x <= 0] = 0
    dR[x > 0] = 1

    return dR

def basic_func(name, x):
    dealer = {
        "sigmoid": Sigmoid,
        "dsigmoid": dSigmoid,
        "tanh": Tanh,
        "dtanh": dTanh,
        "relu": Relu,
        "drelu": dRelu,
    }

    assert (name in dealer.keys())

    return dealer[name](x)