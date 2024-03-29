from scipy.optimize import minimize
import math
import numpy as np

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

path = "../data/data_cauchy.txt"
data = getData(path) 
measurements = MEASUREMENTS(data)

x0, gamma = 50, 100

def objective(x, data):
    x0, gamma = x
    return -sum([math.log(1/(math.pi * gamma * (1 + ((d - x0)/gamma)**2))) for d in measurements.data])

bnds = [(-np.inf, np.inf),(0,np.inf)]
sol = minimize(objective, [46.17,108.45], args = (measurements.data), method="SLSQP", bounds = bnds)
print(sol)