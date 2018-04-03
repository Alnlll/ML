from matplotlib import pyplot as plt

class Ploter(object):
    def __init__(self, linewidth=2):
        self.linewidth = linewidth

    def plot(self, *args, **kwargs):
        plt.plot(*args, **kwargs)

    def axis_label(self, axis=0, label=''):
        if 0 == axis:
            plt.xlabel(label)
        if 1 == axis:
            plt.ylabel(label)

    def draw_line(self, dot1, dot2, set_str = 'k-'):
        plt.figure()
        plt.plot(np.array(([dot1(1),dot2(1)])), np.array(([dot1(2),dot2(2)])), set_str)
        plt.show()

    def show(self):
        plt.show()

import numpy as np

if __name__ == "__main__":

    test = Ploter()
    x = np.array(([1,2]))
    y = x
    test.plot(x,y,linewidth=2)
    test.show()
