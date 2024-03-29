import math

def get_measurements(data: list) -> dict:
    import scipy.stats
    import numpy as np
    measurements = {}
    
    miu_3 = scipy.stats.moment(data, 3)
    miu_4 = scipy.stats.moment(data, 4)
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    skewness = miu_3 / pow(np.std(data, ddof=1),3)
    kurtosis = miu_4 / pow(np.std(data, ddof=1),4)
    median = np.median(data)
    mode = scipy.stats.mode(data)[0][0]
    
    measurements.mean = mean
    measurements.variance =  variance
    measurements.skewness = skewness
    measurements.kurtosis = kurtosis
    measurements.data = data
    measurements.median = median
    measurements.mode = mode
    
    return measurements

def getData(direction):
    file  = open(direction,'r')
    data = [float(x.replace(",",".")) for x in file.read().splitlines()]
    return data

path = "../data/data_generalized_extreme_value.txt"
data = getData(path) 
measurements = MEASUREMENTS(data)

def equations(sol_i, data_mean, data_variance, data_median):
    xi, miu, sigma = sol_i
    
    g = lambda t: math.gamma(1 - t * xi)
    
    parametric_mean = miu + sigma * (g(1) - 1)  / xi
    parametric_variance = (sigma**2) * (g(2) - g(1)**2)  / (xi**2)
    # parametric_skewness = math.copysign(1, xi) * (g(3) - 3*g(2)*g(1) + 2*g(1)**3) / ((g(2)-g(1)**2))**1.5
    # parametric_kurtosis = (g(4) - 4 * g(3) * g(1) - 3*g(2)**2 + 12*g(2)*g(1)**2-6*g(1)**4)/ ((g(2)-g(1)**2))**2
    parametric_median = miu + sigma * (math.log(2) ** (-xi)-1)/xi

    ## System Equations
    eq1 = parametric_mean - data_mean
    eq2 = parametric_variance - data_variance
    eq3 = parametric_median - data_median

    return (eq1, eq2, eq3)

# from scipy.optimize import fsolve
# sol =  fsolve(equations, (1, 1, 1), measurements)
# print(sol)

from scipy.optimize import least_squares
import time
ti = time.time()
res = least_squares(equations, (1, 1, 1), bounds = ((-1, 0, 0), (1, 1000, 1000)), args=(measurements.mean, measurements.variance, measurements.median))
print(res.x)
print(time.time()-ti)
