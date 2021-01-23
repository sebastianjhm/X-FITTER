import scipy.stats
import numpy as np

def get_measurements(data: list) -> dict:
    measurements = {}
    
    miu_3 = scipy.stats.moment(data, 3)
    miu_4 = scipy.stats.moment(data, 4)
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    skewness = miu_3 / pow(np.std(data, ddof=1),3)
    kurtosis = miu_4 / pow(np.std(data, ddof=1),4)
    median = np.median(data)
    
    measurements["mean"] = mean
    measurements["variance"] =  variance
    measurements["skewness"] = skewness
    measurements["kurtosis"] = kurtosis
    measurements["data"] = data
    measurements["median"] = median

    
    return measurements