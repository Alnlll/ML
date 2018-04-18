import numpy as np

# Initialize parameters functions based on randn
def initializer(shape, mode='randn', weight=0.01, seed=1):
    np.random.seed(seed)

    if 'zero' == mode:
        return np.zeros(shape)
    if 'randn' == mode:
        if 1 == len(shape): return np.random.randn(shape[0]) * weight
        if 2 == len(shape): return np.random.randn(shape[0],shape[1]) * weight
        if 3 == len(shape): return np.random.randn(shape[0],shape[1],shape[2]) * weight
        if 4 == len(shape): return np.random.randn(shape[0],shape[1],shape[2],shape[3]) * weight

# Exponentially weighted average function
# v(t) = beta*v(t-1) + (1-beta)*x(t)
def exp_wgt_avg(x_cur, v_prev, beta=0.1, bias_corr=False, t=None):
    if bias_corr: return (beta*v_prev + (1-beta)*x_cur) / (1 - beta**t)
    if not bias_corr: return beta*v_prev + (1-beta)*x_cur

def padder(X, n_pad=1, val=0):
    '''
    Input:
    X -- numpy array of 4-dimension with shape(m,h,w,c)
    n_pad -- integer, amount of padding
    val -- integer, value used to pad

    Return:
    X_pad -- numpy array with shape(m, h+2*pad, w+2*pad, c)
    '''
    return np.pad(X, ((0,0),(n_pad,n_pad),(n_pad,n_pad),(0,0)), 'constant', constant_values=(val))

def pooler(X, f_shape, stride, mode='max'):
    '''
    Input:
    X -- numpy array of 4-dimension with shape(m, n_h, n_w, c)
    f_shape -- tuple, shape of pooler
    stride -- integer, length of step
    mode -- string

    Return:
    X_pool -- numpy array of 4-dimension with shape(m, (n_h-f_h)//stride + 1, (n_w-f_w)//stride + 1, 1)
    '''
    (m, n_h, n_w, n_c) = X.shape
    (f_h, f_w) = f_shape

    max_h = (n_h - f_h) // stride + 1
    max_w = (n_w - f_w) // stride + 1

    X_pool = np.zeros((m, max_h, max_w, n_c))

    for i in range(m):
        for h in range(max_h):
            start_h, end_h = h*stride, h*stride+f_h
            for w in range(max_w):
                start_w, end_w = w*stride, w*stride+f_w
                for c in range(n_c):
                    data_slice = X[i, start_h:end_h, start_w:end_w, c]

                    if 'max' == mode:
                        X_pool[i, h, w, c] = float(np.max(data_slice))
                    if 'average' == mode:
                        X_pool[i, h, w, c] = float(np.mean(data_slice))

    return X_pool

def convolutioner(X, kernel, stride):
    '''
    Input:
    X -- numpy array of 4-dimension with shape(m, n_h, n_w, c)
    kernel -- numpy array, shape (k_h, k_w)
    stride -- integer, length of step

    Return:
    X_conv -- numpy array
    '''
    (m, n_h, n_w, n_c) = X.shape
    (n_kh, n_kw, n_c) = kernel.shape

    max_h = (n_h - n_kh) // stride + 1
    max_w = (n_w - n_kw) // stride + 1

    X_conv = np.zeros((m, max_h, max_w))

    for i in range(m):
        for h in range(max_h):
            start_h, end_h = h*stride, h*stride+n_kh
            for w in range(max_w):
                start_w, end_w = w*stride, w*stride+n_kw

                data_slice = X[i, start_h:end_h, start_w:end_w, :]
                conv_slice = np.sum(kernel * data_slice)
                X_conv[i, h, w] = float(conv_slice)

    return X_conv
