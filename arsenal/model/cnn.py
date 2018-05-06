import numpy as np
from arsenal.common.basic_tool import padder, pooler, convolutioner, initializer
from arsenal.common.basic_func import basic_func

class ConvolutionNeuralNetwork(object):

    def __init__(self, map):
        self.map = map
        self.L = len(map.keys()) - 3

    def initialize(self, end=1, seed=1):
        '''
        Initialize W,b,dW,db,A with zeros or randn.
        Input:
            end -- integer, ignore 'L0', 'FC', the last layer
            seed -- integer, seed for np.random module
        '''

        for l in range(1, end+1):
            params = self.map['L'+str(l)]

            (m, n_h_prev, n_w_prev, n_c_prev) = self.map['L'+str(l-1)]['cache']['A'].shape
            n_fh = params['conv']['n_fh']
            n_fw = params['conv']['n_fw']
            n_c = params['conv']['n_c']
            ops = params['op_list']

            params['cache']['W'] = initializer((n_fh, n_fw, n_c_prev, n_c), mode='randn', weight=1, seed=seed)
            params['cache']['dW'] = initializer((n_fh, n_fw, n_c_prev, n_c), mode='randn', weight=1, seed=seed)
            params['cache']['b'] = initializer((1,1,1,n_c), mode='zero')
            params['cache']['db'] = initializer((1,1,1,n_c), mode='zero')

            n_h, n_w = n_h_prev, n_w_prev

            for op in ops:
                if 'conv' == op:
                    p = params['conv']['pad']
                    stride = params['conv']['stride']
                    n_h = (n_h + 2*p - n_fh) // stride + 1
                    n_w = (n_w + 2*p - n_fw) // stride + 1
                if 'activation' == op:
                    n_h, n_w = n_h, n_w
                if 'pool' == op:
                    stride = params['pool']['stride']
                    n_h = (n_h - n_fh) // stride + 1
                    n_w = (n_w - n_fw) // stride + 1
            params['cache']['A'] = initializer((m, n_h, n_w, n_c), mode='zero')

    def pad(self, X, n_pad=1, val=0):
        return padder(X, n_pad=n_pad, val=val)

    def pool_forward(self, A_prev, f_shape, stride, mode='max'):
        return pooler(A_prev, f_shape, stride, mode=mode)

    def pool_backward(self, dA, A_prev, f_shape, stride, mode='max'):
        '''
        Input:
        dA -- numpy array of shape (m, n_h, n_w, n_c), grad of A by loss
        A_prev -- numpy array of 4-dimension with shape(m, n_h_prev, n_w_prev, n_c_prev)
        f_shape -- tuple, shape of pooler
        stride -- integer, length of step
        mode -- string

        Return:
        dA_prev -- numpy array of 4-dimension with the same shape A_prev
        '''
        (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
        (n_fh, n_fw) = f_shape
        (m, n_h, n_w, n_c) = dA.shape

        dA_prev = np.zeros(A_prev.shape)

        for i in range(m):
            a_prev = A_prev[i, :, :, :]
            for h in range(n_h):
                start_h, end_h = h*stride, h*stride+n_fh
                for w in range(n_w):
                    start_w, end_w = w*stride, w*stride+n_fw
                    for c in range(n_c):
                        a_slice = A_prev[i, start_h:end_h, start_w:end_w, c]

                        if 'max' == mode:
                            mask = (a_slice == np.max(a_slice)) # Assume just one maximum
                            da_prev_slice = mask * dA[i, h, w, c]
                        if 'average' == mode:
                            average =  dA[i, h, w, c] / (n_fh * n_fw)
                            da_prev_slice = average * np.ones((a_slice.shape))

                        dA_prev[i, start_h:end_h, start_w:end_w, c] += da_prev_slice
        return dA_prev

    def convolution_forward(self, A_prev, W, b, stride, p, p_val):
        '''
        Input:
        A_prev -- numpy array, shape of (m, n_h_prev, n_h_prev, n_c_prev)
        W -- numpy array, convolution kernel shape of (m, n_fh, n_fw, n_c)
        b -- numpy array, bias item shape of (n_c, 1)
        stride -- integer, length of convolution step
        p -- integer, amount of padding
        p_val -- integer, value used to pad

        Return:
        Z -- numpy array, convolution of A_prev added by bias item
        '''
        (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
        (n_fh, n_fw, n_c_prev, n_c) = W.shape

        # stride = params['stride']
        # pad_params = params['pad'] # 'pad', 'val'
        # p = pad_params['pad']
        # p_val = pad_params['val']

        n_h = (n_h_prev + 2*p - n_fh) // stride + 1
        n_w = (n_w_prev + 2*p - n_fw) // stride + 1

        Z = np.zeros((m, n_h, n_w, n_c))
        A_prev_pad = self.pad(A_prev, n_pad=p, val=p_val)

        for c in range(n_c):
            Z[:,:,:,c] = convolutioner(A_prev_pad, W[:,:,:,c], stride)
            Z[:,:,:,c] = Z[:,:,:,c] + b[:,:,:,c]

        return Z

    def convolution_backward(self, dZ, A_prev, W, b, stride, p, p_val):

        (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
        (m, n_h, n_w, n_c) = dZ.shape
        (n_fh, n_fw, n_c_prev, n_c) = W.shape

        dA_prev = np.zeros(A_prev.shape)
        dW = np.zeros((W.shape))
        db = np.zeros((1,1,1,n_c))

        A_prev_pad = self.pad(A_prev, n_pad=p, val=p_val)
        dA_prev_pad = self.pad(dA_prev, n_pad=p, val=p_val)

        for i in range(m):
            a_prev_pad = A_prev_pad[i, :, :, :]
            da_prev_pad = dA_prev_pad[i, :, :, :]
            for h in range(n_h):
                start_h, end_h = h*stride, h*stride + n_fh
                for w in range(n_w):
                    start_w, end_w = w*stride, w*stride + n_fw
                    a_prev_slice = a_prev_pad[start_h:end_h, start_w:end_w, :]
                    for c in range(n_c):
                        da_prev_pad[start_h:end_h, start_w:end_w, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_prev_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]

            dA_prev[i,:,:,:] = da_prev_pad[p:-p, p:-p, :]

        return dA_prev, dW, db

    def softmax_forward(self, A_prev):
        return basic_func("softmax", A_prev)

    def softmax_backward(self, dA, A_prev):
        return np.dot(basic_func('dsoftmax', A_prev), dA)

    def fc(self, A_prev):
        return A_prev.flatten()

    def forward_propagation(self, start=1, end=1):
        '''
        Input:
            start -- integer, from which layer to do propagation
            end -- integer, do propagation to which layer
        '''

        assert(0 < start and end <= self.L and start < end)

        for l in range(start, end+1):
            params = self.map['L'+str(l)]
            ops = params['op_list']

            A_prev = self.map['L'+str(l-1)]['cache']['A']

            for op in ops:
                if 'conv' == op:
                    W = params['cache']['W']
                    b = params['cache']['b']
                    stride = params['conv']['stride']
                    p = params['conv']['pad']
                    p_val == params['conv']['val']

                    A_prev = self.convolution_forward(A_prev, W, b, stride, p, p_val)

                if 'activation' == op:
                    activation_name = params['activation']
                    A_prev = basic_func(activation_name, A_prev)

                if 'pool' == op:
                    pool_params = params['pool']
                    stride  = pool_params['stride']
                    f_shape = (pool_params['n_fh'], pool_params['n_fw'])
                    mode = pool_params['mode']

                    A_prev = self.pool(A_prev, f_shape, stride, mode=mode)

            params['cache']['A'] = A_prev
