import h5py
import scipy.io as sio
import numpy as np

class DataHandler(object):
    def __init__(self):
        pass
    def load(self, data_path, delimiter=','):
        
        data_type = data_path.split(".")[-1];
    
        if 'mat' == data_type:
            return sio.loadmat(data_path)

        if data_type in ['csv','txt']:
            return np.loadtxt(data_path, delimiter=delimiter)
            
        if 'h5' == data_type:
            f = h5py.File(data_path)
            keys = [key for key in f.keys()]
            return f,keys

    def convert_to_one_hot(self, X, C):
        return np.eye(C)[X.reshape(-1)].T
            