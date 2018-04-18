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

def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def dSoftmax(x):
    dS = np.zeros((x.shape[0], x.shape[0]))
    S = Softmax(x)

    for i in range(dS.shape[1]):
        dS[:, i] = -S*S[i]
        dS[i, i] = S[i]*(1-S[i])

    return dS

def basic_func(name, x):
    dealer = {
        "sigmoid": Sigmoid,
        "dsigmoid": dSigmoid,
        "tanh": Tanh,
        "dtanh": dTanh,
        "relu": Relu,
        "drelu": dRelu,
        "softmax": Softmax,
        "dsoftmax": dSoftmax,
    }

    assert (name in dealer.keys())

    return dealer[name](x)
