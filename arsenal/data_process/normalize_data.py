import numpy as np

class Normalizer(object):
    def __init__(self):
        pass
    def normalize(self, X, axis = 0):
        mu = np.mean(X, axis = 0, keepdims=True) # axis=0 means by column
        normed = X - mu
        sigma = np.std(normed, axis = 0, keepdims=True, ddof=1) # Divide by (N-ddof)
        normed = normed / sigma

        return mu, sigma, normed
