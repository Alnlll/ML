import numpy as np

def Sigmoid(x):
    '''Calculate Sigmoid value'''
    return 1/(1 + np.exp(-x))