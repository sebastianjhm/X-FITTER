import numpy as np
from scipy.stats import mode, gaussian_kde
from scipy.optimize import minimize, shgo

def generate_1D_data(size, mu=0.5, sigma=1, round=3):
    data = np.random.normal(mu, sigma, size)
    return np.around(data, round)

def kde(array, cut_down=True, bw_method='scott'):
    if cut_down:
        bins, counts = np.unique(array, return_counts=True)
        f_mean = counts.mean()
        f_above_mean = bins[counts > f_mean]
        bounds = [f_above_mean.min(), f_above_mean.max()]
        array = array[np.bitwise_and(bounds[0] < array, array < bounds[1])]
    return gaussian_kde(array, bw_method=bw_method)
 
def mode_estimation(array, cut_down=True, bw_method='scott'):
    kernel = kde(array, cut_down=cut_down, bw_method=bw_method)
    bounds = np.array([[array.min(), array.max()]])
    results = shgo(lambda x: -kernel(x)[0], bounds=bounds, n=100*len(array))
    return results.x[0]

def mode_explicit(array, cut_down=True, bw_method='scott'):
    kernel = kde(array, cut_down=cut_down, bw_method=bw_method)
    height = kernel.pdf(array)
    return array[np.argmax(height)]

def refined_mode_estimation(array, cut_down=True, bw_method='scott'):
    kernel = kde(array, cut_down=cut_down, bw_method=bw_method)
    height = kernel.pdf(array)
    x0 = array[np.argmax(height)]
    span = array.max() - array.min()
    dx = span / 4
    bounds = np.array([[x0 - dx, x0 + dx]])
    linear_constraint = [{'type': 'ineq', 'fun': lambda x:  x - 0.5}]
    results = minimize(lambda x: -kernel(x)[0], x0=x0, bounds=bounds, constraints=linear_constraint)
    return results.x[0]




def getData(direction):
    file  = open(direction,'r')
    data = [float(x.replace(",",".")) for x in file.read().splitlines()]
    return data

path = "C:\\Users\\USUARIO1\\Desktop\\Fitter\\data\\data_triangular.txt"
data = np.array(getData(path))
print(type(data))

mode_ = mode(data)[0][0]
# _mode = mode_explicit(data)
# _mode_ = mode_estimation(data)
_imode_ = refined_mode_estimation(data)

import matplotlib.pyplot as plt

# print(f'actual mode is at {mu}')
print(f'scipy mode: {mode_}')
# print(f'mode_explicit mode: {_mode}')
# print(f'mode_estimation mode: {_mode_}')
print(f'refined_mode_estimation mode: {_imode_}')

bins = np.linspace(data.min(), data.max(), 100)
plt.figure(figsize=(10, 5))
plt.hist(data, bins=100, density=True, label='data', color='b')
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=4, color='r', label='distribution')
plt.axvline(mode_, linewidth=2, label='scipy mode', color='m')
plt.axvline(_mode, linewidth=2, label='explicit mode', color='c')
plt.axvline(_mode_, linewidth=2, label='KDE mode', color='g')
plt.axvline(_imode_, linewidth=2, label='refined mode', color='y')
plt.title('normal distribution with the estimated mode methods')
plt.xlabel('data point values')
plt.ylabel('data point frequencies')
plt.grid()
plt.legend()
plt.show()