import numpy  as np

class KNN(object):
    
    def __init__(self, data, K, centroids):
        self.data = data
        self.K = K
        self.centroids = centroids
        self.assigments = np.zeros((self.data.shape[0], 1))
    
    def distance(self, a, b, axis=0):
        return np.linalg.norm(a - b, axis=axis)
    
    def assign_centroids(self, m):
        for i in range(m):
            dic = self.distance(self.centroids, self.data[i,:], axis=1)
            self.assigments[i] = np.where(np.min(dic) == dic)[0][0]

    def get_valid(self, mask):
        sum = 0
        for i in range(mask.shape[0]):
            if True == mask[i]:
                sum += 1
        return sum

    def update_centroids(self):
        for i in range(self.K):
            mask = (i == self.assigments)
            n_valid = self.get_valid(mask)
            self.centroids[i, :] = np.mean(self.data * mask, axis=0) * (self.assigments.shape[0] / n_valid)
            