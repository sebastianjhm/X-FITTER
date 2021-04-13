import scipy.stats
import numpy as np
from utilities.danoes import danoes_formula

def get_measurements(data: list) -> dict:
    measurements = {}
    
    miu_3 = scipy.stats.moment(data, 3)
    miu_4 = scipy.stats.moment(data, 4)
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    skewness = miu_3 / pow(np.std(data, ddof=1),3)
    kurtosis = miu_4 / pow(np.std(data, ddof=1),4)
    median = np.median(data)
    frequencies, bin_edges = np.histogram(data, danoes_formula(data))
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

if __name__ == "__main__":
    ## Import function to get measurements
    def get_data(direction):
        file  = open(direction,'r')
        data = [float(x.replace(",",".")) for x in file.read().splitlines()]
        return data
    
    ## Distribution class
    path = "..\\data\\data_normal.txt"
    data = get_data(path) 
    
    print(get_measurements(data))