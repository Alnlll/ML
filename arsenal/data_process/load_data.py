import scipy.io as sio

class DataLoader(object):
    def __init__(self, data_path = ''):
        self.data_path = data_path

    def load(self):
        
        data_type = self.data_path.split(".")[-1];
    
        if 'mat' == data_type:
            return sio.loadmat(self.data_path)