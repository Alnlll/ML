from arsenal.common.basic_tool import exp_wgt_avg

class GradDescent(object):
    """GradDescent"""
    def __init__(self):
        self.momentum = {}

    def descent(self, grad, X, reg=None, rate=0.001):
        if reg:
            return X - rate * (grad + reg)
        else:
            return X - rate * grad

    def add_momentum_item(self, name):
        self.momentump[name] = 0

    def descent_momentum(self, grad, X, name,
                         reg=None, rate=0.001,
                         beta=0.9, bias_corr=False, index=None):
        '''
        Descent with exponentially weight average.
        v_dx(t) = beta*v_dx(t-1) + (1-beta)*dx(t)
        x(t) = x(t-1) - rate*(v_dx(t) + reg)
        '''
        try:
            v = exp_wgt_avg(X, self.momentump[name],
                            beta=beta, bias_corr=bias_corr, t=index)
            return self.descent(v, X, reg=reg, rate=rate)
        except Exception as e:
            print("GradDescent::descent_momentum %s" % e)
            return None