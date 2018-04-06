import numpy as np
from arsenal.common.basic_func import Sigmoid, dSigmoid

class LogisticRegression(object):

    def __init__(self):
        pass

    def lgr_func(self, X, w):
        z = np.dot(X, w)
        return Sigmoid(z)

    def lgr_loss(self, X, w, y):
        m = X.shape[0]
        return (1/m) * np.sum(-y*np.log(self.lgr_func(X,w)) - (1-y)*np.log(1-self.lgr_func(X,w)))

    def lgr_grad(self, X, w, y):
        m = X.shape[0]
        return (np.sum((self.lgr_func(X,w) - y)*X, keepdims=True, axis=0) / m).T

    def lgr_loss_grad(self, X, w, y):
        return self.lgr_loss(X, w, y), self.lgr_grad(X, w, y)

    def predict(self, X, w, threashold=0.5):
        y = self.lgr_func(X, w)
        return np.asarray((y >= threashold), dtype=np.int32)