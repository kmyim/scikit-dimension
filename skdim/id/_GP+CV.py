import numpy as np
import skdim
from skdim import _commonfuncs
from skdim._commonfuncs import GlobalEstimator
import pynverse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class GPCV(GlobalEstimator):
    def __init__(self, sample_size=2500, corrint_min_k=10, corrint_max_k=20):
        self.sample_size = sample_size
        self.corrint_min_k = corrint_min_k
        self.corrint_max_k = corrint_max_k
        self.corrint = skdim.id.CorrInt(self.corrint_min_k, self.corrint_max_k)
        self.xdata = []
        self.ydata = []
        self.popt = None

    def generate_data(self, dimensions):
        return skdim.datasets.hyperSphere(self.sample_size, dimensions, random_state=None)

    def fit_corrint(self, data):
        return self.corrint.fit_transform(data)

    def collect_data(self, dim_range):
        for i in dim_range:
            self.xdata.append(i)
            self.ydata.append(self.fit_corrint(self.generate_data(i)))
        self.xdata = np.array(self.xdata)
        self.ydata = np.array(self.ydata)

    def fit_curve(self):
        def model_func(x, a, b, c):
            return a * np.power(x, b) + c

        self.popt, _ = curve_fit(model_func, self.xdata, self.ydata)

    def inverse_function(self, y_values):
        a, b, c = self.popt
        cube = lambda x: a * np.power(x, b) + c
        invcube = pynverse.inversefunc(cube)
        return invcube(y_values)

    def estimate(self, dim_range=range(2, 52)):
        self.collect_data(dim_range)
        self.fit_curve()
        return self

    def transform(self, data):
        corrint_values = self.fit_corrint(data)
        return self.inverse_function(corrint_values)
    
    