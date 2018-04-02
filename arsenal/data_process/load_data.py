import scipy.io as sio
import numpy as np

class DataLoader(object):
    def __init__(self):
        pass
    def load(self, data_path = '', delimiter = ','):
        
        data_type = data_path.split(".")[-1];
    
        if 'mat' == data_type:
            return sio.loadmat(data_path)
        if data_type in ['csv','txt']:
            return np.loadtxt(data_path, delimiter=delimiter)