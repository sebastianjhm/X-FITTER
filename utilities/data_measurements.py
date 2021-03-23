import scipy.stats
import numpy as np
import utilities.danoes as df

def get_measurements(data: list) -> dict:
    measurements = {}
    
    miu_3 = scipy.stats.moment(data, 3)
    miu_4 = scipy.stats.moment(data, 4)
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    skewness = miu_3 / pow(np.std(data, ddof=1),3)
    kurtosis = miu_4 / pow(np.std(data, ddof=1),4)
    median = np.median(data)
    frequencies, bin_edges = np.histogram(data, df.danoes_formula(data))
    i = list(frequencies).index(max(frequencies))
    mode = (bin_edges[i]+bin_edges[i+1])/2
    
    measurements["mean"] = mean
    measurements["variance"] =  variance
    measurements["skewness"] = skewness
    measurements["kurtosis"] = kurtosis
    measurements["data"] = data
    measurements["median"] = median
    measurements["mode"] = mode
    
    return measurements