class GradDescent(object):
    """GradDescent"""
    def __init__(self):
        pass
    def descent(self, grad, X, reg=None, rate = 0.001):
        if reg:
            return X - rate * (grad + reg)
        else:
            return X - rate * grad
        