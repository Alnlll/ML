class GradDescent(object):
    """GradDescent"""
    def __init__(self):
        pass
    def descent(self, grad, X, rate = 0.001):
        return X - rate * grad
        