from matplotlib import pyplot as plt

class Ploter(object):
    def __init__(self, linewidth=2):
        self.linewidth = linewidth

    def plot(self, data1, data2, set_str = 'r+'):
        plt.figure()
        plt.plot(data1, data2, set_str, linewidth=self.linewidth)
        plt.show()

    def draw_line(self, dot1, dot2, set_str = 'k-'):
        plt.figure()
        plt.plot(np.array(([dot1(1),dot2(1)])), np.array(([dot1(2),dot2(2)])), set_str)
        plt.show()
